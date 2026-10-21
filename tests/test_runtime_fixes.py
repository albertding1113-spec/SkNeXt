"""Focused runtime regressions, isolated to keep GUI imports dependency-light."""
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACT = '''
import ast
from pathlib import Path
def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
'''


class RuntimeFixTests(unittest.TestCase):
    def run_probe(self, code):
        result = subprocess.run([sys.executable, "-c", EXTRACT + textwrap.dedent(code)],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_augmentation_accepts_nested_and_section_configs(self):
        self.run_probe('''
            import numpy as np
            from sknext.data.augmentor import PairedAugmentor
            from yacs.config import CfgNode
            section = {"ENABLE": True, "DA_PROB": 1., "HFLIP": True}
            raw = np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1)
            target = (raw > 10).astype(np.uint8)
            for cfg in (section, {"AUGMENTOR": section}, CfgNode(section), CfgNode({"AUGMENTOR": section})):
                x, y = PairedAugmentor(cfg)(raw, target)
                assert np.array_equal(x, np.flip(raw, 2))
                assert np.array_equal(y, np.flip(target, 2))
            x, y = PairedAugmentor({"AUGMENTOR": {**section, "ENABLE": False}})(raw, target)
            assert np.array_equal(x, raw) and np.array_equal(y, target)
        ''')

    def test_model_stage_kernels_do_not_leak(self):
        self.run_probe('''
            import torch
            from torch import nn
            from typing import Dict, List
            # Isolate model construction from the installed torchvision binary.
            # These stand-ins capture block arguments; no forward pass is claimed.
            class Permute(nn.Module):
                def __init__(self, dims):
                    super().__init__()
            class ConvNeXtBlock_V2(nn.Module):
                def __init__(self, *args, k_size, **kwargs):
                    super().__init__()
                    self.kernel = k_size
            class UpConvNeXtBlock_V2(ConvNeXtBlock_V2):
                pass
            init_weights = lambda model: None
            definitions("sknext/model/unext_v2.py", {"U_NeXt_V2"}, globals())
            for flags in ([False, True, True], [True, False, True], [True, True, False]):
                model = U_NeXt_V2(image_shape=(8, 16, 16, 1), feature_maps=[4, 8, 16],
                                 z_down=[1, 2], yx_down=[2, 2], cn_layers=[1, 1, 1],
                                 isotropy=flags, output_channels=[1, 1],
                                 output_channel_info=["F", "P"], separated_decoders=True)
                expected = [(7, 7, 7) if f else (1, 7, 7) for f in flags]
                assert [model.down_path[i + 1][0].kernel for i in range(2)] == expected[:2]
                assert model.bottleneck[0].kernel == expected[-1]
                for decoder in model.up_paths:
                    assert [block.kernel for block in decoder[:-1]] == expected[-2::-1]
        ''')

    def test_semantic_resume_rejects_changed_prediction_settings(self):
        self.run_probe('''
            import copy, json, math, os, tempfile
            import numpy as np
            import torch
            names = {"cfgget", "checked_semantic_names", "get_thresholds", "spatial_patch",
                     "overlap_voxels", "normalize_options", "checkpoint_path", "checkpoint_read_path",
                     "_resolved_output", "infer"}
            definitions("sknext/run/semantic_workflow.py", names, globals())
            from types import SimpleNamespace
            make_model = lambda cfg, device: SimpleNamespace(eval=lambda: None)
            load_model_checkpoint = lambda *args: None
            predictions = []
            def predict(*args):
                predictions.append(1)
                return np.ones((1, 4, 4, 4), dtype=np.float32)
            _predict_block = predict
            class Reader:
                shape = (1, 4, 4, 4)
                def close(self): pass
                def __getitem__(self, key): return np.ones((4, 4, 4), dtype=np.uint16)
            _input_reader = lambda *args: (Reader(), [0])
            class Writer:
                def __init__(self, path, **kwargs):
                    Path(path).mkdir(exist_ok=True)
                    self.root = SimpleNamespace(attrs={})
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def __setitem__(self, key, value): pass
            ZarrIOManager = Writer
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                checkpoint = base / "model.pth"
                checkpoint.touch()
                cfg = {"TASK": {"CHANNELS": ["S.test"]},
                       "MODEL": {"CHECKPOINT_INPUT_PATH": str(checkpoint)},
                       "DATA": {"PATCH_SIZE": [4, 4, 4, 1],
                                "INFER": {"PATH": str(base / "input.zarr"),
                                          "GT_PATH": str(base / "out.zarr"),
                                          "BLOCK_CENTRAL_FACTOR": [1, 1, 1]}}}
                infer(cfg, 0, torch.device("cpu"))
                infer(cfg, 0, torch.device("cpu"))
                assert len(predictions) == 1  # unchanged resume skips completed block
                for section, values in (
                    ("DATA", {"NORMALIZATION": {"TYPE": "zero_mean_unit_variance"}}),
                    ("INFER", {"SKIP_ZERO_PATCHES": False}),
                ):
                    changed = copy.deepcopy(cfg)
                    changed.setdefault(section, {}).update(values)
                    try: infer(changed, 0, torch.device("cpu"))
                    except RuntimeError as exc: assert "Cannot resume" in str(exc)
                    else: raise AssertionError("Changed prediction settings accepted")
                changed = copy.deepcopy(cfg)
                changed["DATA"]["INFER"]["OVERLAP"] = [0.5, 0.5, 0.5]
                try: infer(changed, 0, torch.device("cpu"))
                except RuntimeError as exc: assert "Cannot resume" in str(exc)
                else: raise AssertionError("Changed overlap accepted")
                progress = base / "out.zarr.progress.json"
                legacy = json.loads(progress.read_text())
                legacy["meta"].pop("normalization")
                progress.write_text(json.dumps(legacy))
                try: infer(cfg, 0, torch.device("cpu"))
                except RuntimeError as exc: assert "Cannot resume" in str(exc)
                else: raise AssertionError("Unverifiable legacy progress accepted")
        ''')
