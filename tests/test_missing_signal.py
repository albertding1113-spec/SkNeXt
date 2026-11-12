"""Missing-signal intensity handling shared by all model workflows."""
import unittest
from unittest.mock import patch

import numpy as np

from sknext.data.augmentor import PairedAugmentor, cutout
from sknext.data.preprocessing import fill_inference_zeros


class MissingSignalTests(unittest.TestCase):
    def test_fill_nonzero_percentile_per_channel_without_mutation(self):
        for dtype in (np.uint8, np.uint16, '>u2'):
            raw = np.array([[[[0, 10, 20, 30]]], [[[0, 100, 200, 0]]]], dtype=dtype)
            original = raw.copy()
            result = fill_inference_zeros(raw)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result[:, 0, 0, 0], [10.2, 101.])
            np.testing.assert_array_equal(result[raw != 0], raw[raw != 0])
            np.testing.assert_array_equal(raw, original)

    def test_empty_constant_and_float_channels(self):
        raw = np.array([[[[0, 0, 0]]], [[[0, 7, 7]]]], dtype=np.uint16)
        np.testing.assert_array_equal(fill_inference_zeros(raw),
                                      [[[[0, 0, 0]]], [[[7, 7, 7]]]])
        for dtype in (np.float32, np.int16, np.uint32):
            raw = np.array([[[[0, 1, 2]]]], dtype=dtype)
            np.testing.assert_array_equal(fill_inference_zeros(raw), raw)

    def test_cutout_range_controls_intensity_for_both_training_tasks(self):
        raw = np.linspace(0, 1, 1000, dtype=np.float32).reshape(10, 10, 10, 1)
        target = np.ones_like(raw)
        for task in ('segmentation', 'sr'):
            aug = PairedAugmentor({'CUT_OUT': True, 'CUT_OUT_INTENSITY_RANGE': (0.01, 0.01)}, task=task)
            with patch.object(aug, '_enabled', side_effect=lambda key, *args: key == 'CUT_OUT'):
                result, labels = aug(raw, target)
            changed = result != raw
            self.assertTrue(changed.any())
            np.testing.assert_allclose(result[changed], .01)
            if task == 'sr':
                np.testing.assert_array_equal(labels, target)
            else:
                np.testing.assert_array_equal(labels[changed], 0)

    def test_cutout_range_default_and_validation(self):
        self.assertEqual(PairedAugmentor({}).cut_out_intensity_range, (0., .05))
        self.assertEqual(PairedAugmentor({}).cut_out_size, .30)
        for interval in ((-.1, .1), (.2, .1), (0, 2), (0,), (0, np.nan)):
            with self.assertRaisesRegex(ValueError, 'CUT_OUT_INTENSITY_RANGE'):
                PairedAugmentor({'CUT_OUT_INTENSITY_RANGE': interval})
        for size in (0, -.1, 1.1, np.nan):
            with self.assertRaisesRegex(ValueError, 'CUT_OUT_SIZE'):
                PairedAugmentor({'CUT_OUT_SIZE': size})

    def test_cutout_size_limits_region_in_both_tasks(self):
        raw = np.linspace(0, 1, 100**3, dtype=np.float32).reshape(100, 100, 100, 1)
        for task in ('segmentation', 'sr'):
            for maximum in (.02, .10):
                aug = PairedAugmentor({'CUT_OUT': True, 'CUT_OUT_SIZE': maximum,
                                       'CUT_OUT_INTENSITY_RANGE': (.01, .01)}, task=task)
                with patch.object(aug, '_enabled', side_effect=lambda key, *args: key == 'CUT_OUT'):
                    result, _ = aug(raw, np.ones_like(raw))
                coords = np.argwhere(np.any(result != raw, axis=-1))
                extents = np.ptp(coords, axis=0) + 1
                self.assertTrue(np.all(extents <= round(100 * maximum)))
                self.assertTrue(np.all(extents >= round(100 * min(.05, maximum))))

    def test_edge_and_slab_cover_requested_faces_in_2d_and_3d(self):
        for shape in ((12, 16), (8, 12, 16)):
            raw = np.arange(np.prod(shape) * 2, dtype=np.float32).reshape(*shape, 2)
            original = raw.copy()
            for geometry in ('edge', 'slab'):
                for face in range(2 * len(shape)):
                    for channel_mask in (False, True):
                        mask = np.ones((*shape, 2) if channel_mask else shape, dtype=np.uint8)
                        with patch('numpy.random.choice', return_value=face):
                            result, labels = cutout(raw, mask, values=(.01, .01),
                                                    size=(.25, .25), geometry=geometry)
                        missing = labels[..., 0] == 0 if channel_mask else labels == 0
                        coords = np.argwhere(missing)
                        lower, upper = coords.min(axis=0), coords.max(axis=0) + 1
                        axis, high_end = divmod(face, 2)
                        self.assertEqual(upper[axis] if high_end else lower[axis],
                                         shape[axis] if high_end else 0)
                        expected = np.array(shape) // 4
                        if geometry == 'slab':
                            expected = np.array(shape)
                            expected[axis] //= 4
                        np.testing.assert_array_equal(upper - lower, expected)
                        self.assertEqual(missing.sum(), np.prod(expected))
                        np.testing.assert_array_equal(result[~missing], raw[~missing])
                        for channel in range(2):
                            np.testing.assert_allclose(result[..., channel][missing],
                                                       np.percentile(raw[..., channel], 1))
                        np.testing.assert_array_equal(raw, original)
                        self.assertTrue(np.all(mask == 1))

    def test_mixed_geometry_routes_through_both_training_tasks(self):
        raw = np.linspace(0, 1, 8**3, dtype=np.float32).reshape(8, 8, 8, 1)
        for task in ('segmentation', 'sr'):
            for geometry in ('interior', 'edge', 'slab'):
                aug = PairedAugmentor({'CUT_OUT': True, 'CUT_OUT_GEOMETRY': 'mixed',
                                       'CUT_OUT_SIZE': .25,
                                       'CUT_OUT_INTENSITY_RANGE': (.01, .01)}, task=task)
                choices = [geometry] if geometry == 'interior' else [geometry, 0]
                with patch.object(aug, '_enabled', side_effect=lambda key, *args: key == 'CUT_OUT'), \
                        patch('numpy.random.choice', side_effect=choices):
                    result, labels = aug(raw, np.ones_like(raw))
                changed = result != raw
                self.assertTrue(changed.any())
                np.testing.assert_allclose(result[changed], .01)
                if task == 'sr':
                    self.assertTrue(np.all(labels == 1))
                else:
                    np.testing.assert_array_equal(labels == 0, changed)

    def test_geometry_validation_and_small_patches(self):
        with self.assertRaisesRegex(ValueError, 'CUT_OUT_GEOMETRY'):
            PairedAugmentor({'CUT_OUT_GEOMETRY': 'unknown'})
        with self.assertRaisesRegex(ValueError, 'geometry'):
            cutout(np.ones((2, 2, 1)), geometry='unknown')
        for geometry in ('interior', 'edge', 'slab', 'mixed'):
            raw = np.ones((1, 1, 1, 1), dtype=np.float32)
            result, labels = cutout(raw, raw, geometry=geometry)
            np.testing.assert_array_equal(result, raw)
            np.testing.assert_array_equal(labels, np.zeros_like(raw))


if __name__ == '__main__':
    unittest.main()
