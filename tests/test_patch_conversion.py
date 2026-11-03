"""Serial/process parity with real TIFF input and Zarr output."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

import numpy as np
import tifffile
import zarr

from sknext.data.datasetIO import tif_list_to_zarr
from sknext.data.patch_workers import bounded_map, conversion_pool, resolve_conversion_workers


class PatchConversionTests(unittest.TestCase):
    def test_process_output_matches_serial(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            raws, labels, semantics = [], [], []
            for i, shape in enumerate(((5, 9, 13), (2, 3, 3))):
                raw = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
                label = np.zeros(shape, dtype=np.uint16)
                label[:, 1:3, 1:3] = 1
                for paths, suffix, data in ((raws, '', raw), (labels, '_gt', label),
                                            (semantics, '_gt.cell', label)):
                    path = base / f'sample{i}{suffix}.tif'
                    tifffile.imwrite(path, data, photometric='minisblack')
                    paths.append(path)
            for normalize, filtered in ((True, False), (False, True)):
                stores = []
                for workers in (1, 2):
                    output = base / f'{normalize}-{workers}.ome.zarr'
                    tif_list_to_zarr(
                        raws[::-1], {'instance': labels, 'S.cell': semantics}, output,
                        (4, 6, 6, 1), (0, 1, 1), (0, 0, 0),
                        preprocess_dict={'normalization': True, 'norm_type': 'scale_range'},
                        channels=['F', 'P', 'A', 'S.cell'],
                        channel_extra_opts={'P': {'type': 'skeleton', 'skeleton_mode': 'main'}},
                        filter_dict={'enable': filtered, 'props': ['label_mean'],
                                     'values': [0], 'signs': ['gt']},
                        normalize_per_patch=normalize, conversion_workers=workers)
                    stores.append(zarr.open_group(str(output), mode='r'))
                for key in ('raw', 'label'):
                    np.testing.assert_array_equal(stores[0][key][:], stores[1][key][:])
                self.assertEqual(dict(stores[0].attrs), dict(stores[1].attrs))

    def test_invalid_workers_do_not_remove_store(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / 'existing.zarr'
            output.mkdir()
            for workers in (-1, 1.5, True):
                with self.assertRaises(ValueError):
                    tif_list_to_zarr([], {}, output, (4, 4, 4, 1), (0, 0, 0),
                                     (0, 0, 0), conversion_workers=workers)
                self.assertTrue(output.exists())
        self.assertTrue(1 <= resolve_conversion_workers(0) <= 8)

    def test_submission_is_bounded_and_pending_jobs_cancelled(self):
        pool = Mock()
        pool.submit.side_effect = lambda fn, job: Mock(result=lambda: job)
        results = bounded_map(pool, abs, iter(range(100)), 2)
        self.assertEqual(next(results), 0)
        self.assertEqual(pool.submit.call_count, 4)
        results.close()

    def test_spawned_worker_failure_propagates(self):
        with conversion_pool(2) as pool:
            with self.assertRaises(ValueError):
                list(bounded_map(pool, int, ['not-an-integer'], 2))


if __name__ == '__main__':
    unittest.main()
