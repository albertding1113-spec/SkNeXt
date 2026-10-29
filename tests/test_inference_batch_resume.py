"""Verify inference manifests independently of GPU and volume I/O dependencies."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


class ModelConfig(dict):
    def clone(self):
        return self.copy()


class InferenceBatchResumeTests(unittest.TestCase):
    def test_sr_output_location_and_input_protection(self):
        node = method('sknext/run/restoration_workflow.py', 'RestorationWorkflow', '_output_paths')
        ns = dict(Path=Path)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'sr_output', 'exec'), ns)
        resolve = ns['_output_paths']
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            base = Path(folder)
            source = base / 'sample.tif'
            source.touch()
            infer = SimpleNamespace(PATH=str(source), GT_PATH=str(base / 'nested' / 'outputs'))
            workflow = SimpleNamespace(cfg=SimpleNamespace(DATA=SimpleNamespace(INFER=infer)), scale=2)
            output, progress = resolve(workflow, source)
            self.assertEqual(output, base / 'nested' / 'outputs' / 'sample_SR_2x.ome.zarr')
            self.assertEqual(progress, output.with_name('sample_SR_2x.progress.json'))
            for name, log in [('custom.ome.zarr', 'custom.progress.json'),
                              ('custom.zarr', 'custom.progress.json')]:
                infer.GT_PATH = str(base / name)
                self.assertEqual(resolve(workflow, source), (base / name, base / log))
            inputs = base / 'raw'
            inputs.mkdir()
            infer.PATH = str(inputs)
            with self.assertRaisesRegex(ValueError, 'output directory'):
                resolve(workflow, source)
            infer.GT_PATH = str(base / 'outputs')
            self.assertEqual(resolve(workflow, inputs / 'other.tif')[0],
                             base / 'outputs' / 'other_SR_2x.ome.zarr')
            source = base / 'input.zarr'
            source.mkdir()
            infer.PATH = str(source)
            infer.GT_PATH = str(base / 'custom.zarr')
            self.assertEqual(resolve(workflow, source)[0], base / 'custom.zarr')
            for target in (source, source / 'nested.zarr'):
                infer.GT_PATH = str(target)
                with self.assertRaisesRegex(ValueError, 'separate'):
                    resolve(workflow, source)
            infer.GT_PATH = ''
            with self.assertRaisesRegex(ValueError, 'GT_PATH'):
                resolve(workflow, source)

    def test_segmentation_manifest_ignores_batch_size(self):
        node = method('sknext/run/networkflow.py', 'Segmentation_Workflow', '_build_infer_manifest')
        ns = dict(Path=Path, hashlib=hashlib)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'manifest', 'exec'), ns)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            source = Path(folder) / 'source.ims'
            source.write_bytes(b'input')
            checkpoint = Path(folder) / 'weights.pth'
            checkpoint.write_bytes(b'weights')
            cfg = SimpleNamespace(
                MODEL=ModelConfig(ARCHITECTURE='test'), TASK={'CHANNELS': ['S.ER']},
                DATA=SimpleNamespace(NORMALIZATION={'ENABLE': False}),
                TRAIN=SimpleNamespace(BATCH_SIZE=1),
                INFER=SimpleNamespace(STITCH_MIN_OVERLAP_VOXELS=1, STITCH_MIN_FRACTION=.5))
            workflow = SimpleNamespace(
                cfg=cfg, batch_size=1, infer_path=source, checkpoint_input_file=checkpoint,
                infer_reader=SimpleNamespace(shape=(1, 8, 16, 16)),
                _file_identity=lambda p: {'path': str(p), 'size': p.stat().st_size},
                skeleton_flag=False, job_id=7, output_mode='uint16',
                save_segmentation=True, save_raw_predictions=False,
                infer_gt_path=Path(folder) / 'output.zarr', infer_channel=[0],
                patch_size=[4, 8, 8, 1], infer_overlap=[0, 0, 0], infer_padding=[1, 1, 1],
                block_size=[8, 16, 16], c_block_size=[4, 8, 8],
                watershed_seed_channels=[], watershed_seed_channels_thresh=[],
                watershed_topographic_channel=0, watershed_growth_mask_channels=[],
                watershed_growth_mask_channels_thresh=[], postprocess_dict={},
                raw_all_blocks=True, output_channel_names=['S.ER'])
            build = ns['_build_infer_manifest']
            original = build(workflow)
            cfg.TRAIN.BATCH_SIZE = workflow.batch_size = 8
            self.assertEqual(build(workflow), original)
            workflow.infer_padding = [0, 0, 0]
            self.assertNotEqual(build(workflow), original)

    def test_sr_manifest_ignores_batch_size(self):
        node = method('sknext/run/restoration_workflow.py', 'RestorationWorkflow', '_infer_one')
        # Execute the production manifest construction without opening a volume or model.
        assignments = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id in {'manifest_args', 'manifest'}
                               for t in n.targets)]
        code = compile(ast.Module(body=assignments, type_ignores=[]), 'sr_manifest', 'exec')
        cfg = SimpleNamespace(TRAIN=SimpleNamespace(BATCH_SIZE=1),
                              SR=SimpleNamespace(RAW_RANGE=[], GT_RANGE=[], OUTPUT_CLIP=True))
        workflow = SimpleNamespace(cfg=cfg, scale=2, patch=[4, 8, 8])
        ns = dict(self=workflow, source=ROOT / 'source.ims', ckpt=ROOT / 'weights.pth',
                  stat=SimpleNamespace(st_size=100, st_mtime_ns=1),
                  ckptstat=SimpleNamespace(st_size=200, st_mtime_ns=2),
                  input_bounds=(0., 1.), physical_ch=[0], shape=(8, 16, 16),
                  padding=(1, 1, 1), central_block=(4, 8, 8), block_margin=(1, 1, 1),
                  geometry={'block_size': (8, 16, 16)}, hashlib=hashlib, json=json)
        exec(code, ns)
        original = ns['manifest']
        cfg.TRAIN.BATCH_SIZE = 8
        exec(code, ns)
        self.assertEqual(ns['manifest'], original)
        ns['padding'] = (0, 0, 0)
        exec(code, ns)
        self.assertNotEqual(ns['manifest'], original)


if __name__ == '__main__':
    unittest.main()
