"""
Activation extraction via forward hooks.

Provides two complementary paths for extracting activations:
  Path A: Hidden states via output_hidden_states=True (residual stream)
  Path B: Forward hooks on named modules (e.g., model.layers[i].mlp.down_proj)
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Optional

import torch
from torch import Tensor, nn


class ActivationHookManager:
    """
    Manages forward hooks on specified model modules to capture activations.

    Usage:
        manager = ActivationHookManager(model)

        # Register hooks on specific modules
        manager.register_hooks([
            "model.layers.10.mlp.down_proj",
            "model.layers.14.mlp.down_proj",
        ])

        # Run forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Retrieve captured activations
        activations = manager.get_activations()
        # activations["model.layers.10.mlp.down_proj"] -> Tensor[batch, seq, hidden]

        # Clean up
        manager.remove_hooks()
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list = []
        self._activations: dict[str, Tensor] = {}

    def register_hooks(
        self,
        module_names: list[str],
        hook_fn: Optional[Callable] = None,
    ) -> None:
        """
        Register forward hooks on the named modules.

        Args:
            module_names: List of module paths (e.g., "model.layers.10.mlp.down_proj")
            hook_fn: Optional custom hook function. If None, uses default (stores output).
        """
        self.remove_hooks()  # Clean up any existing hooks
        self._activations.clear()

        for name in module_names:
            module = self._get_module(name)
            if module is None:
                raise ValueError(f"Module '{name}' not found in model")

            if hook_fn:
                hook = module.register_forward_hook(hook_fn)
            else:
                hook = module.register_forward_hook(self._make_hook(name))

            self._hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        """Create a hook function that stores the module output."""

        def hook_fn(module, input, output):
            # Handle both Tensor and tuple outputs
            if isinstance(output, tuple):
                output = output[0]
            # Store as float32 for numerical stability in probe training
            self._activations[name] = output.detach().float()

        return hook_fn

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Navigate the model's module tree to find a named module."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, "__getitem__"):
                module = module[int(part)]
            else:
                return None
        return module

    def get_activations(self) -> dict[str, Tensor]:
        """Return captured activations from the most recent forward pass."""
        return dict(self._activations)

    def clear_activations(self) -> None:
        """Clear captured activations (free memory)."""
        self._activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


@contextmanager
def capture_activations(model: nn.Module, module_names: list[str]):
    """
    Context manager for activation capture.

    Usage:
        with capture_activations(model, ["model.layers.14.mlp.down_proj"]) as manager:
            outputs = model(**inputs)
            acts = manager.get_activations()
    """
    manager = ActivationHookManager(model)
    manager.register_hooks(module_names)
    try:
        yield manager
    finally:
        manager.remove_hooks()


def get_hidden_states(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Optional[Tensor] = None,
    layers: Optional[list[int]] = None,
) -> dict[int, Tensor]:
    """
    Path A: Extract hidden states from the residual stream.

    Args:
        model: The language model.
        input_ids: Input token IDs [batch, seq_len].
        attention_mask: Optional attention mask.
        layers: Which layers to return (None = all).

    Returns:
        Dict mapping layer index to activation tensor [batch, seq_len, hidden_size].
        Layer 0 = embedding output, Layer N = final layer output.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors

    result = {}
    for i, hs in enumerate(hidden_states):
        if layers is None or i in layers:
            result[i] = hs.detach().float()

    return result


def extract_features(
    activations: Tensor,
    method: str = "last_token",
    k: int = 8,
) -> Tensor:
    """
    Extract a fixed-size feature vector from variable-length activations.

    Args:
        activations: Tensor of shape [batch, seq_len, hidden_size]
        method: Feature extraction method:
            - "last_token": activation at the last token position
            - "mean_last_k": mean over last k token positions
            - "mean_all": mean over all token positions
            - "first_token": activation at position 0
        k: Number of tokens for "mean_last_k" method.

    Returns:
        Feature tensor of shape [batch, hidden_size].
    """
    if method == "last_token":
        return activations[:, -1, :]
    elif method == "mean_last_k":
        return activations[:, -k:, :].mean(dim=1)
    elif method == "mean_all":
        return activations.mean(dim=1)
    elif method == "first_token":
        return activations[:, 0, :]
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")


def residual_stream_module_names(num_layers: int = 28) -> list[str]:
    """Return standard hook point names for all residual stream positions in Qwen2."""
    names = []
    for i in range(num_layers):
        names.extend([
            f"model.layers.{i}",  # Full layer output (residual stream)
            f"model.layers.{i}.self_attn",  # Attention output
            f"model.layers.{i}.mlp",  # MLP output
            f"model.layers.{i}.mlp.down_proj",  # MLP down projection
            f"model.layers.{i}.mlp.gate_proj",  # MLP gate projection
            f"model.layers.{i}.mlp.up_proj",  # MLP up projection
        ])
    return names


def get_probe_hook_points(
    layers: list[int],
    modules: list[str] | None = None,
) -> list[str]:
    """
    Generate hook point names for probing.

    Args:
        layers: Which layer indices to probe.
        modules: Which submodules per layer. Defaults to ["mlp.down_proj"].

    Returns:
        List of full module path strings.
    """
    if modules is None:
        modules = ["mlp.down_proj"]

    return [f"model.layers.{layer}.{module}" for layer in layers for module in modules]
