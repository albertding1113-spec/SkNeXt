"""Test conversion decisions without importing the GPU workflow dependencies."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class PatchStoreReuseTests(unittest.TestCase):
    def test_sr_reuse_validates_before_conversion(self):
        source = Path(__file__).resolve().parents[1] / 'sknext/run/restoration_workflow.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'validate_sr_patch_store', 'make_sr_patch_store'}]
        cfg = SimpleNamespace(
            TRAIN=SimpleNamespace(REUSE_PATCH_STORES=True),
            DATA=SimpleNamespace(PATCH_SIZE=[2, 4, 4, 1]),
            SR=SimpleNamespace(UPSAMPLE=2, RAW_RANGE=[], GT_RANGE=[]))
        class Root(dict):
            attrs = {'sknext': dict(task='SR', normalization='per_image_minmax_v1',
                                    raw_range=[], gt_range=[])}
        root = Root(raw=SimpleNamespace(shape=(3, 1, 2, 4, 4)),
                    label=SimpleNamespace(shape=(3, 1, 2, 8, 8)))
        ns = dict(Path=Path, validate_sr=lambda cfg: 2, log_message=Mock(),
                  zarr=SimpleNamespace(open_group=Mock(return_value=root)))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.assertEqual(ns['make_sr_patch_store'](cfg, 'TRAIN', path), path)
            root['label'].shape = (2, 1, 2, 8, 8)
            with self.assertRaisesRegex(ValueError, 'invalid paired NCZYX'):
                ns['make_sr_patch_store'](cfg, 'TRAIN', path)
            del root['label']
            with self.assertRaisesRegex(ValueError, 'REUSE_PATCH_STORES: False'):
                ns['make_sr_patch_store'](cfg, 'TRAIN', path)

    def test_conversion_policy(self):
        source = Path(__file__).resolve().parents[1] / 'sknext/run/networkflow.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'Segmentation_Workflow')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'save_patch_as_zarr')
        for reuse, existing, expected in (
            (True, ('train', 'val'), []),
            (True, ('train',), ['val']),
            (True, ('val',), ['train']),
            (True, (), ['train', 'val']),
            (False, ('train', 'val'), ['train', 'val']),
        ):
            with self.subTest(reuse=reuse, existing=existing), TemporaryDirectory() as tmp:
                root = Path(tmp)
                for split in existing:
                    store = root / f'{split}.ome.zarr'
                    store.mkdir()
                    (store / 'sentinel').write_text('unchanged')
                convert = Mock()
                ns = dict(Path=Path, time_str=lambda: '',
                          get_cfg_value=lambda cfg, key, default: cfg.get(key, default),
                          get_tif_path_in_folder=lambda path: [path],
                          get_tif_path_dict_in_folder=lambda path, channels: {'F': [path]},
                          pair_raw_and_label_tiffs=lambda raw, labels: (raw, labels),
                          tif_list_to_zarr=convert)
                exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
                runner = SimpleNamespace(
                    cfg={'TRAIN.REUSE_PATCH_STORES': True} if reuse else {},
                    train_path=root / 'train_raw', val_path=root / 'val_raw',
                    train_gt_path=root / 'train_gt', val_gt_path=root / 'val_gt',
                    channels=['F'], patch_size=(8, 8, 8),
                    train_overlap=0, val_overlap=0, train_padding=True, val_padding=True,
                    preprocess_dict={}, channels_extra_opts={}, filter_dict={'enabled': True})
                ns['save_patch_as_zarr'](runner)
                self.assertEqual([c.args[2].name for c in convert.call_args_list],
                                 [f'{split}.ome.zarr' for split in expected])
                for call, split in zip(convert.call_args_list, expected):
                    self.assertEqual(call.kwargs['filter_dict'],
                                     runner.filter_dict if split == 'train' else None)
                    self.assertTrue(call.kwargs['normalize_per_patch'])
                for split in existing:
                    self.assertEqual((root / f'{split}.ome.zarr/sentinel').read_text(), 'unchanged')


if __name__ == '__main__':
    unittest.main()
