import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sknext.config.config import SkNeXt_Config, load_config, update_config
from sknext.gui import discover_gpus, parse_value, validate


class LauncherTests(unittest.TestCase):
    """Regression checks for GUI configuration handling and workflow preflight."""
    def test_import_does_not_load_training_runtime(self):
        """Verify importing the editor leaves PyTorch and the workflow runtime unloaded."""
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("sknext._sknext", sys.modules)

    def test_typed_fields(self):
        """Check scalar/sequence parsing and reject incompatible or executable expressions."""
        self.assertEqual(parse_value("[20, 32, 32, 1]", (1,)), (20, 32, 32, 1))
        self.assertEqual(parse_value("5e-4", 0.0), 0.0005)
        self.assertEqual(parse_value("D:/data with spaces", ""), "D:/data with spaces")
        for value in ("True", "1.5", "__import__('os').getcwd()"):
            with self.assertRaises(ValueError):
                parse_value(value, 1)

    def test_gpu_uuid_and_memory(self):
        """Check that NVIDIA query output becomes a memory label mapped to a GPU UUID."""
        with patch("sknext.gui.subprocess.run") as run:
            run.return_value.stdout = "2, GPU-abc, NVIDIA Example, 100, 200\n"
            self.assertEqual(discover_gpus(), {"GPU 2 · NVIDIA Example · 100/200 MiB free": "GPU-abc"})

    def test_yaml_round_trip_recomputes_derived_paths(self):
        """Verify saved settings survive YAML reload and derived paths follow the new run ID."""
        cfg = SkNeXt_Config(0).get_cfg_defaults()
        cfg.PATHS.RESULT_DIR.PATH = "experiment with spaces"
        cfg.DATA.PATCH_SIZE = (16, 64, 64, 1)
        cfg.TRAIN.LR = 0.0002
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.yaml"
            path.write_text(cfg.dump(), encoding="utf-8")
            restored = update_config(SkNeXt_Config(7).get_cfg_defaults(), load_config(str(path)), 7)
            self.assertEqual(tuple(restored.DATA.PATCH_SIZE), (16, 64, 64, 1))
            self.assertEqual(restored.TRAIN.LR, 0.0002)
            self.assertEqual(Path(restored.PATHS.CHECKPOINT_DIR), Path("experiment with spaces/results/7/checkpoints"))

    def test_training_and_inference_preflight(self):
        """Check training paths, overlap validation, and the inference checkpoint requirement."""
        with tempfile.TemporaryDirectory() as folder:
            workdir = Path(folder)
            for name in ("raw", "label", "skeleton"):
                (workdir / name).mkdir()
            cfg = update_config(SkNeXt_Config(3).get_cfg_defaults(), SkNeXt_Config(3).get_cfg_defaults(), 3)
            cfg.TRAIN.ENABLE = True
            validate(cfg, 3, workdir)
            cfg.DATA.TRAIN.OVERLAP = (1, 0, 0)
            with self.assertRaisesRegex(ValueError, "Overlap"):
                validate(cfg, 3, workdir)
            cfg.DATA.TRAIN.OVERLAP = (0, 0, 0)
            cfg.TRAIN.ENABLE = False
            cfg.INFER.ENABLE = True
            with self.assertRaisesRegex(ValueError, "Checkpoint"):
                validate(cfg, 3, workdir)
            checkpoint = workdir / cfg.PATHS.CHECKPOINT_DIR / "checkpoint_03.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            validate(cfg, 3, workdir)


if __name__ == "__main__":
    unittest.main()
