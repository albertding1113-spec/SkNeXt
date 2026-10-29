"""Selection and batching regressions, runnable without a CUDA installation."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from sknext.utils.parallel import (configure_visible_devices, inference_batch_size,
                                   parallel_model, model_state_dict)


class ParallelTests(unittest.TestCase):
    def test_visibility_and_all_preserve_scheduler_mask(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_visible_devices('all')
            self.assertNotIn('CUDA_VISIBLE_DEVICES', os.environ)
            configure_visible_devices('2, 0,1')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2,0,1')
            configure_visible_devices('all')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2,0,1')

    def test_invalid_selection_does_not_change_visibility(self):
        for selector in ('', '0,', '-1', '0,0', 'abc', 'all,0'):
            with self.subTest(selector=selector), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
                with self.assertRaises(ValueError):
                    configure_visible_devices(selector)
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '2')

    def test_parallel_device_order_and_small_batch_warning(self):
        model = Mock()
        model.to.return_value = model
        torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 3),
                                nn=SimpleNamespace(DataParallel=Mock(return_value='wrapped')))
        with patch.dict('sys.modules', {'torch': torch}), self.assertWarns(UserWarning):
            result = parallel_model(model, SimpleNamespace(type='cuda', index=2), 1)
        self.assertEqual(result, 'wrapped')
        torch.nn.DataParallel.assert_called_once_with(model, device_ids=[2, 0, 1], output_device=2)

    def test_inference_batches_and_portable_weights(self):
        model = SimpleNamespace(device_ids=[0, 1, 2], module=SimpleNamespace(state_dict=lambda: {'weight': 7}))
        self.assertEqual(inference_batch_size(model, 1), 3)
        self.assertEqual(inference_batch_size(model, 8), 8)
        self.assertEqual(inference_batch_size(object(), 1), 1)
        self.assertEqual(model_state_dict(model), {'weight': 7})

    def test_cuda_outputs_gradients_and_checkpoint_match_single_gpu(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('Requires at least two CUDA GPUs')
        from sknext.utils.checkpoint import load_model_weights
        torch.manual_seed(42)
        device = torch.device('cuda:0')
        reference = torch.nn.Conv3d(1, 2, 1).to(device)
        parallel = parallel_model(torch.nn.Conv3d(1, 2, 1), device)
        load_model_weights(parallel, reference.state_dict())
        # An uneven batch verifies gather order and partial replica batches.
        x = torch.randn(2 * torch.cuda.device_count() + 1, 1, 3, 3, 3, device=device)
        expected, actual = reference(x), parallel(x)
        torch.testing.assert_close(actual, expected)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for a, b in zip(reference.parameters(), parallel.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        restored = torch.nn.Conv3d(1, 2, 1).to(device)
        load_model_weights(restored, model_state_dict(parallel))
        torch.testing.assert_close(restored(x), expected)
