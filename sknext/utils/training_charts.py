"""Persistent epoch metrics and loss charts for standalone training workflows."""
from __future__ import annotations

import json
from pathlib import Path


class TrainingLossChart:
    """Keep a JSONL history and refresh a train/validation PNG after each epoch."""

    def __init__(self, history_file: Path, chart_file: Path, start_epoch: int = 0):
        self.history_file = Path(history_file)
        self.chart_file = Path(chart_file)
        self.history = {}
        if start_epoch > 0 and self.history_file.exists():
            for line in self.history_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                epoch = int(record["epoch"])
                if epoch <= start_epoch:
                    self.history[epoch] = record
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.chart_file.parent.mkdir(parents=True, exist_ok=True)
        # Discard metrics newer than the checkpoint when resuming, or reset a new run.
        with self.history_file.open("w", encoding="utf-8") as stream:
            for epoch in sorted(self.history):
                stream.write(json.dumps(self.history[epoch]) + "\n")
        self.chart_file.unlink(missing_ok=True)
        if self.history:
            self.save()

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        record = {"epoch": int(epoch), **{key: float(value) for key, value in metrics.items()}}
        with self.history_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        self.history[int(epoch)] = record
        self.save()

    def save(self) -> None:
        # Use an explicit noninteractive canvas, including on headless training hosts.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(9, 6))
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        epochs = sorted(self.history)
        for name in ("train_loss", "val_loss"):
            ax.plot(epochs, [self.history[epoch][name] for epoch in epochs],
                    marker="o", markersize=3, linewidth=1.5, label=name)
        ax.set_title(f"Training loss (through epoch {epochs[-1]})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        figure.tight_layout()
        figure.savefig(self.chart_file, dpi=200, bbox_inches="tight")
