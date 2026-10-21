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
from sknext.utils.progress import log_message
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

    def test_sr_normalizes_each_integer_and_float_image_independently(self):
        self.run_probe('''
            import numpy as np
            from typing import Sequence
            definitions('sknext/run/restoration_workflow.py',
                        {'image_intensity_range', 'normalize_intensity'}, globals())
            integer = np.array([0, 32768, 65535], dtype=np.uint16)
            floating = np.array([-.032258, .680591, 1.39344], dtype=np.float32)
            for image in (integer, floating):
                result = normalize_intensity(image, [], 'raw')
                assert result.dtype == np.float32
                np.testing.assert_allclose(result, [0., .5, 1.], atol=1e-5)
            # Calling normalization on uint16 data must never affect another image.
            before = normalize_intensity(floating, [], 'raw')
            normalize_intensity(integer, [], 'raw')
            np.testing.assert_array_equal(normalize_intensity(floating, [], 'raw'), before)
            for image in (np.zeros(4, dtype=np.uint16), np.full(4, 5., dtype=np.float32)):
                np.testing.assert_array_equal(normalize_intensity(image, [], 'raw'), np.zeros(4))
            np.testing.assert_allclose(normalize_intensity(np.array([0., 1.]), [0., 2.], 'raw'), [0., .5])
            for image in (np.array([0., np.nan]), np.array([0., np.inf]), np.array([])):
                try: normalize_intensity(image, [], 'raw')
                except ValueError: pass
                else: raise AssertionError('Invalid input accepted')
            try: normalize_intensity(np.array([3.]), [0., 2.], 'raw')
            except ValueError: pass
            else: raise AssertionError('Explicit bounds ignored')
        ''')

    def test_sr_patch_store_normalizes_before_cropping_mixed_images(self):
        self.run_probe('''
            import numpy as np
            import tempfile
            from types import SimpleNamespace
            from typing import Sequence
            names = {'image_intensity_range', 'normalize_intensity', 'make_sr_patch_store',
                     'align_sr_gt', '_overlap_voxels', 'crop_padded'}
            definitions('sknext/run/restoration_workflow.py', names, globals())
            images = {'integer': np.array([0, 16384, 32768, 65535], dtype=np.uint16).reshape(1,1,1,4),
                      'float': np.array([-.03, .3, .7, 1.39344], dtype=np.float32).reshape(1,1,1,4),
                      'gt1': np.array([0., 1., 2., 4.], dtype=np.float32).reshape(1,1,1,4),
                      'gt2': np.array([0, 100, 200, 400], dtype=np.uint16).reshape(1,1,1,4)}
            pair_sr_tiffs = lambda *args: [(Path('integer'),Path('gt1')), (Path('float'),Path('gt2'))]
            read_one_3D_tif = lambda path, axes: images[str(path)].copy()
            validate_sr = lambda cfg: 1
            patch_coordinates_iter = lambda *args: [np.array([[0,1],[0,1],[0,2]]),
                                                      np.array([[0,1],[0,1],[2,4]])]
            class Array:
                def __init__(self): self.data = {}
                def resize(self, shape): pass
                def __setitem__(self, key, value): self.data[key] = value.copy()
            class Root:
                def __init__(self): self.attrs = {}; self.arrays = {}
                def create_array(self, name, **kwargs):
                    self.arrays[name] = Array()
                    return self.arrays[name]
            root = Root()
            zarr = SimpleNamespace(open_group=lambda *args, **kwargs: root)
            BloscCodec = lambda **kwargs: None
            cfg = SimpleNamespace(SR=SimpleNamespace(RAW_RANGE=[], GT_RANGE=[], RESCALE_GT=False),
                DATA=SimpleNamespace(PATCH_SIZE=[1,1,2,1],
                                     TRAIN=SimpleNamespace(PATH='raw', GT_PATH='gt', OVERLAP=[0,0,0])))
            with tempfile.TemporaryDirectory() as folder:
                make_sr_patch_store(cfg, 'TRAIN', Path(folder)/'patches.zarr')
            assert cfg.SR.RAW_RANGE == [] and cfg.SR.GT_RANGE == []
            for i, name in enumerate(('integer', 'float')):
                combined = np.concatenate([root.arrays['raw'].data[2*i],root.arrays['raw'].data[2*i+1]],axis=-1)
                np.testing.assert_allclose(combined, normalize_intensity(images[name], [], 'raw'))
                gtname = ('gt1','gt2')[i]
                combined_gt = np.concatenate([root.arrays['label'].data[2*i],root.arrays['label'].data[2*i+1]],axis=-1)
                np.testing.assert_allclose(combined_gt, normalize_intensity(images[gtname], [], 'GT'))
            assert root.attrs['sknext']['normalization'] == 'per_image_minmax_v1'
            assert len(root.attrs['sknext']['image_ranges']) == 2
        ''')

    def test_sr_inference_uses_one_range_per_source_and_rejects_old_policy(self):
        self.run_probe('''
            import numpy as np
            from types import SimpleNamespace
            from typing import Sequence
            definitions('sknext/run/restoration_workflow.py',
                        {'image_intensity_range', 'normalize_intensity', 'reader_intensity_range'}, globals())
            class Reader:
                def __init__(self):
                    self.data = np.stack([np.linspace(-.03,1.39344,8).reshape(2,2,2),
                                          np.full((2,2,2),65535.)])
                    self.shape = self.data.shape
                    self.reads = 0
                def __getitem__(self, key):
                    self.reads += 1
                    return self.data[key]
            reader = Reader()
            bounds = reader_intensity_range(reader, [0], (1,1,1), [])
            np.testing.assert_allclose(bounds, [-.03,1.39344])
            assert reader.reads == 8
            whole = normalize_intensity(reader.data[0], [], 'raw')
            for z in range(2):
                block = normalize_intensity(reader.data[0,z:z+1], bounds, 'raw')
                np.testing.assert_allclose(block, whole[z:z+1])
            before = reader.reads
            assert reader_intensity_range(reader, [0], (1,1,1), [0.,2.]) == (0.,2.)
            assert reader.reads == before
            tree = ast.parse(Path('sknext/run/restoration_workflow.py').read_text(encoding='utf-8'))
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='RestorationWorkflow')
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='_validate_intensity_policy')
            exec(compile(ast.Module(body=[node],type_ignores=[]),'policy','exec'),globals())
            _validate_intensity_policy(None, {'normalization_policy':'per_image_minmax_v1'})
            try: _validate_intensity_policy(None, {'raw_range':[-.03,65535.]})
            except ValueError as exc: assert 'fresh run' in str(exc)
            else: raise AssertionError('Old shared ranges silently accepted')
        ''')

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
            from sknext.utils.model_info import print_cuda_memory
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

    def test_gpu_memory_report_and_cpu_guard(self):
        self.run_probe('''
            import contextlib, io
            from types import SimpleNamespace
            from unittest.mock import Mock
            class Device:
                def __init__(self, name):
                    self.name = name
                    self.type = name.split(':')[0]
                def __str__(self): return self.name
            cuda = Mock()
            cuda.is_available.return_value = True
            mib = 1024 * 1024
            cuda.memory_allocated.return_value = mib
            cuda.memory_reserved.return_value = 2 * mib
            cuda.max_memory_allocated.return_value = 3 * mib
            cuda.max_memory_reserved.return_value = 4 * mib
            cuda.mem_get_info.return_value = (6 * mib, 8 * mib)
            torch = SimpleNamespace(device=Device, cuda=cuda)
            definitions('sknext/utils/model_info.py', {'bytes_to_mb', 'print_cuda_memory'}, globals())
            print_cuda_memory(Device('cpu'))
            cuda.is_available.assert_not_called()
            cuda.synchronize.assert_not_called()
            output = io.StringIO()
            device = Device('cuda:1')
            with contextlib.redirect_stdout(output):
                print_cuda_memory(device, prefix='test phase')
            text = output.getvalue()
            assert 'test phase' in text and 'cuda:1' in text
            expected = {'Allocated': 1, 'Reserved': 2, 'Max allocated': 3,
                        'Max reserved': 4, 'Device used': 2, 'Free memory': 6, 'Total memory': 8}
            for name, value in expected.items():
                lines = [line.split('] ', 1)[1] for line in text.splitlines()]
                line = next(line for line in lines if line.startswith(name + ':'))
                assert line.split(':', 1)[1].strip() == f'{value:.2f} MiB'
            cuda.synchronize.assert_called_once_with(device)
            cuda.mem_get_info.assert_called_once_with(device)
            cuda.is_available.return_value = False
            cuda.synchronize.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()): print_cuda_memory()
            cuda.synchronize.assert_not_called()
        ''')
