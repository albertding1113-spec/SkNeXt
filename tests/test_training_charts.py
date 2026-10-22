"""Verify loss chart output across fresh and resumed training runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib import image as mpimg
from matplotlib.figure import Figure

from sknext.utils.training_charts import TrainingLossChart


class TrainingLossChartTests(unittest.TestCase):
    def test_losses_are_plotted_and_saved_with_other_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            chart = TrainingLossChart(base / "history.jsonl", base / "chart" / "Loss.png")
            chart.log(1, {"train_loss": 0.8, "val_loss": 0.9, "val_psnr_db": 20.0})
            chart.log(2, {"train_loss": 0.5, "val_loss": 0.6, "val_psnr_db": 22.0})
            records = [json.loads(line) for line in chart.history_file.read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [1, 2])
            self.assertEqual(records[-1]["val_psnr_db"], 22.0)
            with patch.object(Figure, "savefig", autospec=True) as savefig:
                chart.save()
            axes = savefig.call_args.args[0].axes[0]
            self.assertEqual(axes.get_xlabel(), "Epoch")
            self.assertEqual([line.get_label() for line in axes.lines], ["train_loss", "val_loss"])
            self.assertEqual(list(axes.lines[0].get_xdata()), [1, 2])
            self.assertEqual(list(axes.lines[0].get_ydata()), [0.8, 0.5])
            self.assertEqual(list(axes.lines[1].get_ydata()), [0.9, 0.6])
            pixels = mpimg.imread(chart.chart_file)
            self.assertGreater(pixels.shape[0], 100)
            self.assertGreater(float(pixels.std()), 0.01)

    def test_resume_discards_epochs_after_checkpoint_and_fresh_run_resets(self):
        with tempfile.TemporaryDirectory() as folder:
            history = Path(folder) / "history.jsonl"
            png = Path(folder) / "Loss.png"
            chart = TrainingLossChart(history, png)
            for epoch in (1, 2, 3):
                chart.log(epoch, {"train_loss": 1 / epoch, "val_loss": 2 / epoch})
            resumed = TrainingLossChart(history, png, start_epoch=2)
            self.assertEqual(sorted(resumed.history), [1, 2])
            resumed.log(3, {"train_loss": 0.1, "val_loss": 0.2})
            records = [json.loads(line) for line in history.read_text().splitlines()]
            self.assertEqual([r["epoch"] for r in records], [1, 2, 3])
            self.assertEqual(records[-1]["val_loss"], 0.2)
            TrainingLossChart(history, png)
            self.assertEqual(history.read_text(), "")
            self.assertFalse(png.exists())

    def test_resume_from_legacy_checkpoint_without_history(self):
        with tempfile.TemporaryDirectory() as folder:
            chart = TrainingLossChart(Path(folder) / "history.jsonl",
                                      Path(folder) / "Loss.png", start_epoch=4)
            chart.log(5, {"train_loss": 0.3, "val_loss": 0.4})
            self.assertEqual(list(chart.history), [5])
            self.assertTrue(chart.chart_file.is_file())
