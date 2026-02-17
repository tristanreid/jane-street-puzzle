"""
Logging and experiment tracking utilities.

Provides consistent logging for activation extraction, probe training,
and trigger search experiments.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

console = Console()


class ExperimentLogger:
    """
    Simple experiment logger that writes structured JSON logs.

    Usage:
        logger = ExperimentLogger("probe_sweep_001")
        logger.log_params({"layers": [10, 14], "method": "contrast_pair"})
        logger.log_metric("auroc", 0.95, step=0)
        logger.log_artifact("probe_direction.npy", "data/probe_models/direction.npy")
        logger.finish()
    """

    def __init__(self, experiment_name: str, base_dir: str = "data/results"):
        self.name = experiment_name
        self.start_time = time.time()
        self.timestamp = datetime.now().isoformat()

        self.log_dir = Path(base_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_entries: list[dict] = []
        self.params: dict = {}
        self.metrics: dict[str, list] = {}
        self.artifacts: list[dict] = []

        console.print(f"[bold green]Experiment:[/bold green] {experiment_name}")
        console.print(f"  Log dir: {self.log_dir}")

    def log_params(self, params: dict) -> None:
        """Log experiment parameters."""
        self.params.update(params)
        self._append_entry("params", params)

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if name not in self.metrics:
            self.metrics[name] = []
        entry = {"value": value, "step": step, "time": time.time() - self.start_time}
        self.metrics[name].append(entry)
        self._append_entry("metric", {"name": name, **entry})

    def log_artifact(self, name: str, path: str) -> None:
        """Log an artifact (file path)."""
        self.artifacts.append({"name": name, "path": path})
        self._append_entry("artifact", {"name": name, "path": path})

    def log_text(self, key: str, text: str) -> None:
        """Log free-form text."""
        self._append_entry("text", {"key": key, "text": text})

    def _append_entry(self, entry_type: str, data: dict) -> None:
        self.log_entries.append({
            "type": entry_type,
            "timestamp": time.time() - self.start_time,
            "data": data,
        })

    def finish(self) -> Path:
        """Finalize and save the experiment log."""
        elapsed = time.time() - self.start_time

        summary = {
            "experiment_name": self.name,
            "start_time": self.timestamp,
            "elapsed_seconds": elapsed,
            "params": self.params,
            "final_metrics": {k: v[-1]["value"] if v else None for k, v in self.metrics.items()},
            "artifacts": self.artifacts,
            "log_entries": self.log_entries,
        }

        log_path = self.log_dir / "experiment_log.json"
        with open(log_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        console.print(f"\n[bold green]Experiment complete:[/bold green] {elapsed:.1f}s")
        console.print(f"  Log saved to: {log_path}")
        return log_path


def print_probe_results(results: list[dict], title: str = "Probe Results") -> None:
    """Pretty-print probe evaluation results as a rich table."""
    table = Table(title=title)
    table.add_column("Layer", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("AUROC", style="green", justify="right")
    table.add_column("AUPRC", style="yellow", justify="right")
    table.add_column("N+", justify="right")
    table.add_column("N-", justify="right")

    for r in results:
        table.add_row(
            str(r.get("layer", "?")),
            r.get("method", "?"),
            f"{r.get('auroc', 0):.4f}",
            f"{r.get('auprc', 0):.4f}",
            str(r.get("n_positive", "?")),
            str(r.get("n_negative", "?")),
        )

    console.print(table)


def save_generation_log(
    prompts: list[str],
    responses: list[str],
    metadata: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a structured log of model generations for analysis."""
    log = {
        "metadata": metadata,
        "generations": [
            {"prompt": p, "response": r}
            for p, r in zip(prompts, responses)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)
