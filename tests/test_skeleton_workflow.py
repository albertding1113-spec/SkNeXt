"""Synthetic-volume regressions for skeleton-associated organelle export."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import navis
import numpy as np
import tifffile

from sknext.run.skeletonworkflow import Skeleton_Workflow


class Reader:
    def __init__(self, volume):
        self.volume = volume
        self.shape = volume.shape
        self.closed = False

    def __getitem__(self, key):
        return self.volume[key]

    def close(self):
        self.closed = True


class SkeletonWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.swcs = self.root / 'swcs'
        self.swcs.mkdir()
        (self.swcs / '0042.swc').write_text(
            '1 1 2 6 6 1 -1\n2 2 6 6 6 1 1\n3 2 10 6 6 1 2\n')

    def workflow(self, volume, **kwargs):
        self.reader = Reader(volume)
        settings = dict(channel=[0], min_values=[1], subregion=[[-1]],
                        central_block=(6, 6, 6), block_halo=(5, 5, 5))
        settings.update(kwargs)
        with patch.object(Skeleton_Workflow, 'create_reader',
                          lambda obj: setattr(obj, 'reader', self.reader)):
            return Skeleton_Workflow(self.swcs, 'unused.zarr', self.root / 'out', **settings)

    def test_export_once_across_blocks_and_roundtrip(self):
        volume = np.zeros((1, 14, 14, 14), dtype=np.uint8)
        volume[0, 5:8, 5:8, 5:8] = 1
        workflow = self.workflow(volume)
        workflow.run()
        self.assertTrue(self.reader.closed)
        files = list((self.root / 'out').rglob('data.json'))
        self.assertEqual(len(files), 1)
        metadata = json.loads(files[0].read_text())
        self.assertEqual(metadata['min_coord'], [[5, 8]] * 3)
        self.assertEqual(metadata['voxel_count'], 27)
        self.assertEqual(metadata['swc_file_id'], 42)
        np.testing.assert_array_equal(tifffile.imread(files[0].with_suffix('.tif')),
                                      np.ones((3, 3, 3), dtype=np.uint8))
        nodes = navis.read_swc(files[0].with_suffix('.swc')).nodes
        self.assertEqual(int((nodes.parent_id < 0).sum()), 1)
        self.assertEqual(int(nodes.loc[nodes.parent_id < 0, 'label'].iloc[0]), 1)
        tip = nodes.loc[~nodes.node_id.isin(nodes.parent_id)].iloc[0]
        np.testing.assert_allclose(tip[['z', 'y', 'x']].to_numpy(dtype=float), [6, 6, 6])

    def test_filter_size_compartment_and_boundary(self):
        volume = np.zeros((1, 14, 14, 14), dtype=np.uint8)
        volume[0, 5:8, 5:8, 5:8] = 1
        for kwargs in ({'min_values': [28]}, {'subregion': [[4]]}):
            self.workflow(volume, **kwargs).run()
            self.assertFalse(list((self.root / 'out').rglob('data.json')))
        volume[0, :6, 6, 6] = 1
        self.workflow(volume, central_block=(14, 14, 14)).run()
        self.assertFalse(list((self.root / 'out').rglob('data.json')))

    def test_equal_bounding_minima_do_not_overwrite(self):
        volume = np.zeros((1, 14, 14, 14), dtype=np.uint8)
        # Two disconnected L shapes have the same ZYX bounding-box minima.
        volume[0, 5, 5, 5:8] = 1
        volume[0, 5, 5:8, 5] = 1
        volume[0, 5, 5:10, 9] = 1
        volume[0, 5, 9, 5:10] = 1
        workflow = self.workflow(volume)
        bounds = np.array([[0, 14]] * 3)
        workflow.calc_save_one_block(volume[0], np.ones(volume.shape[1:], dtype=np.uint16),
                                     0, bounds, bounds)
        self.assertEqual(len(list((self.root / 'out').rglob('data.tif'))), 2)

    def test_errors_propagate_and_reader_closes(self):
        workflow = self.workflow(np.ones((1, 14, 14, 14), dtype=np.uint8))
        with patch.object(workflow.skeleton, 'crop_skeletons', side_effect=RuntimeError('broken')):
            with self.assertRaisesRegex(RuntimeError, 'broken'):
                workflow.run()
        self.assertTrue(self.reader.closed)

    def test_invalid_settings_and_channel(self):
        volume = np.zeros((1, 14, 14, 14), dtype=np.uint8)
        for kwargs in ({'channel': [0, 0]}, {'central_block': (0, 2, 2)},
                       {'connectivity': 4}, {'min_values': []}, {'subregion': []}):
            with self.assertRaises(ValueError):
                self.workflow(volume, **kwargs)
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.workflow(volume, channel=[1])
        self.assertTrue(self.reader.closed)


if __name__ == '__main__':
    unittest.main()
