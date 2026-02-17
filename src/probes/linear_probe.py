"""
Linear probe training and evaluation for backdoor detection.

Implements the methodology from "Simple Probes Can Catch Sleeper Agents"
(MacDiarmid et al., 2024):
  1. Compute probe direction from contrast pair activations
  2. Project test activations onto probe direction → scalar score
  3. Evaluate with AUROC, AUPRC, and calibration metrics

Also supports scikit-learn based probes (LogisticRegression, LinearSVC, Ridge).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class ProbeResult:
    """Results from probe evaluation."""

    auroc: float
    auprc: float
    scores: NDArray  # Raw probe scores for all test examples
    labels: NDArray  # True labels for all test examples
    fpr: NDArray  # False positive rates (for ROC curve)
    tpr: NDArray  # True positive rates (for ROC curve)
    thresholds: NDArray  # Thresholds (for ROC curve)
    metadata: dict  # Additional info (layer, method, etc.)

    def to_dict(self) -> dict:
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "n_positive": int(self.labels.sum()),
            "n_negative": int(len(self.labels) - self.labels.sum()),
            **self.metadata,
        }


class LinearProbe:
    """
    Simple linear probe using the contrast-pair direction method.

    This is the primary method from Anthropic's probes paper:
    direction = mean(positive_activations) - mean(negative_activations)
    score = activation @ direction

    No training loop needed — just compute the direction from contrast pairs.
    """

    def __init__(self):
        self.direction: Optional[NDArray] = None
        self.bias: float = 0.0
        self.metadata: dict = {}

    def fit(
        self,
        positive_activations: NDArray,
        negative_activations: NDArray,
        normalize: bool = True,
    ) -> "LinearProbe":
        """
        Compute the probe direction from contrast pair activations.

        Args:
            positive_activations: Activations for "positive" examples [N_pos, hidden_size]
                (e.g., "I am doing something hidden" → yes)
            negative_activations: Activations for "negative" examples [N_neg, hidden_size]
                (e.g., "I am doing something hidden" → no)
            normalize: Whether to L2-normalize the direction vector.

        Returns:
            self (for chaining).
        """
        pos_mean = positive_activations.mean(axis=0)
        neg_mean = negative_activations.mean(axis=0)

        self.direction = pos_mean - neg_mean

        if normalize:
            norm = np.linalg.norm(self.direction)
            if norm > 1e-10:
                self.direction = self.direction / norm

        # Set bias to the midpoint of the two means projected onto the direction
        pos_score = pos_mean @ self.direction
        neg_score = neg_mean @ self.direction
        self.bias = (pos_score + neg_score) / 2

        self.metadata["fit_n_positive"] = len(positive_activations)
        self.metadata["fit_n_negative"] = len(negative_activations)
        self.metadata["fit_method"] = "contrast_pair_direction"

        return self

    def score(self, activations: NDArray) -> NDArray:
        """
        Score activations by projecting onto the probe direction.

        Args:
            activations: Activation vectors [N, hidden_size]

        Returns:
            Scalar scores [N], higher = more "positive" (triggered/defecting).
        """
        if self.direction is None:
            raise RuntimeError("Probe has not been fitted yet")
        return activations @ self.direction - self.bias

    def predict(self, activations: NDArray, threshold: float = 0.0) -> NDArray:
        """Binary prediction based on threshold."""
        return (self.score(activations) > threshold).astype(int)

    def save(self, path: Path) -> None:
        """Save probe direction and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "direction.npy", self.direction)
        with open(path / "metadata.json", "w") as f:
            json.dump({"bias": self.bias, **self.metadata}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LinearProbe":
        """Load a saved probe."""
        path = Path(path)
        probe = cls()
        probe.direction = np.load(path / "direction.npy")
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        probe.bias = meta.pop("bias")
        probe.metadata = meta
        return probe


def train_probe(
    train_features: NDArray,
    train_labels: NDArray,
    method: str = "logistic",
    **kwargs,
) -> object:
    """
    Train a scikit-learn linear probe.

    Args:
        train_features: Feature vectors [N, D].
        train_labels: Binary labels [N].
        method: One of "logistic", "svm", "ridge".
        **kwargs: Passed to the sklearn estimator.

    Returns:
        Fitted sklearn estimator.
    """
    # Scale features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_features)

    if method == "logistic":
        clf = LogisticRegression(max_iter=1000, **kwargs)
    elif method == "svm":
        clf = LinearSVC(max_iter=1000, **kwargs)
    elif method == "ridge":
        clf = RidgeClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    clf.fit(X_scaled, train_labels)

    # Attach the scaler for use at inference time
    clf._feature_scaler = scaler
    return clf


def evaluate_probe(
    probe,
    test_features: NDArray,
    test_labels: NDArray,
    metadata: Optional[dict] = None,
) -> ProbeResult:
    """
    Evaluate a probe (LinearProbe or sklearn estimator) on test data.

    Args:
        probe: A LinearProbe or fitted sklearn estimator.
        test_features: Feature vectors [N, D].
        test_labels: True binary labels [N].
        metadata: Optional dict of experiment metadata.

    Returns:
        ProbeResult with AUROC, AUPRC, scores, etc.
    """
    if metadata is None:
        metadata = {}

    # Get scores
    if isinstance(probe, LinearProbe):
        scores = probe.score(test_features)
    elif hasattr(probe, "decision_function"):
        X_scaled = probe._feature_scaler.transform(test_features)
        scores = probe.decision_function(X_scaled)
    elif hasattr(probe, "predict_proba"):
        X_scaled = probe._feature_scaler.transform(test_features)
        scores = probe.predict_proba(X_scaled)[:, 1]
    else:
        raise ValueError(f"Unsupported probe type: {type(probe)}")

    # Compute metrics
    auroc = roc_auc_score(test_labels, scores)
    auprc = average_precision_score(test_labels, scores)
    fpr, tpr, thresholds = roc_curve(test_labels, scores)

    return ProbeResult(
        auroc=auroc,
        auprc=auprc,
        scores=scores,
        labels=test_labels,
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        metadata=metadata,
    )


def probe_sweep(
    activations_by_layer: dict[str, NDArray],
    labels: NDArray,
    methods: list[str] = ["contrast_pair", "logistic"],
    test_split: float = 0.3,
    random_state: int = 42,
) -> list[ProbeResult]:
    """
    Run probes across multiple layers and methods.

    Args:
        activations_by_layer: Dict mapping layer/module name to features [N, D].
        labels: Binary labels [N].
        methods: List of probe methods to try.
        test_split: Fraction for test set.
        random_state: Random seed.

    Returns:
        List of ProbeResult objects sorted by AUROC (descending).
    """
    from sklearn.model_selection import train_test_split

    results = []

    for layer_name, features in activations_by_layer.items():
        # Split data
        idx = np.arange(len(features))
        train_idx, test_idx = train_test_split(
            idx, test_size=test_split, random_state=random_state, stratify=labels
        )

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        for method in methods:
            meta = {"layer": layer_name, "method": method}

            if method == "contrast_pair":
                probe = LinearProbe()
                pos_mask = y_train == 1
                neg_mask = y_train == 0
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue
                probe.fit(X_train[pos_mask], X_train[neg_mask])
                result = evaluate_probe(probe, X_test, y_test, metadata=meta)
            else:
                clf = train_probe(X_train, y_train, method=method)
                result = evaluate_probe(clf, X_test, y_test, metadata=meta)

            results.append(result)

    # Sort by AUROC descending
    results.sort(key=lambda r: r.auroc, reverse=True)
    return results
