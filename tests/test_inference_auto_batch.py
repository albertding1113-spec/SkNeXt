"""Inference probe behavior and workflow routing without requiring CUDA."""
from contextlib import ExitStack
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sknext.utils.auto_batch import determine_inference_batch_size, _inference_trial
from sknext.utils.parallel import resolve_inference_batch


class InferenceAutoBatchTests(unittest.TestCase):
    def test_semantic_runtime_oom_retries_without_losing_or_double_blending_patches(self):
        source = Path(__file__).resolve().parents[1] / 'sknext/run/semantic_workflow.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = {'_predict_block', '_semantic_batch_probabilities', '_prediction_model_result', '_window'}
        ns = dict(np=np, torch=torch,
                  spatial_patch=lambda cfg: (1, 1, 1),
                  overlap_voxels=lambda *args: (0, 0, 0),
                  patch_coordinates_iter=lambda *args: [((0, 1), (0, 1), (i, i + 1)) for i in range(7)],
                  normalize_options=lambda cfg: {},
                  cfgget=lambda cfg, key, default: cfg.get(key, default),
                  reflect_padding_img=lambda raw, size: raw,
                  preprocess_img=lambda raw, norm: raw,
                  log_message=lambda *args, **kwargs: None)
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in names], type_ignores=[]), str(source), 'exec'), ns)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
                self.limit = 2

            def forward(self, x):
                self.calls.append(len(x))
                if len(x) > self.limit:
                    raise torch.cuda.OutOfMemoryError('simulated runtime OOM')
                return x

        model = Model()
        model._sknext_inference_batch = ('key', 4)
        raw = np.arange(7, dtype=np.float32).reshape(1, 1, 1, 7)
        cfg = {'INFER.AUTO_BATCH_SIZE': True}
        with patch('sknext.utils.parallel.resolve_inference_batch',
                   side_effect=lambda *args, **kwargs: model._sknext_inference_batch[1]), \
                patch('torch.cuda.empty_cache'):
            result = ns['_predict_block'](model, raw, cfg, torch.device('cpu'), ['a'], False)
            np.testing.assert_allclose(result, torch.sigmoid(torch.from_numpy(raw)).numpy()
                                       * (raw != 0), rtol=1e-6)
            self.assertEqual(model._sknext_inference_batch, ('key', 1))
            model.calls.clear()
            ns['_predict_block'](model, raw, cfg, torch.device('cpu'), ['a'], False)
            self.assertEqual(model.calls, [1] * 6)
            model.limit = 0
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                ns['_predict_block'](model, raw, cfg, torch.device('cpu'), ['a'], False)
            model._sknext_inference_batch = ('key', 4)
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                ns['_predict_block'](model, raw, {}, torch.device('cpu'), ['a'], False)

    def test_forward_only_for_every_workflow(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                assert not torch.is_grad_enabled()
                return {'pred': x * 2}
        for workflow in ('semantic', 'segmentation', 'restoration'):
            with self.subTest(workflow=workflow):
                _inference_trial(Model(), (1, 2, 3, 4), 2, torch.device('cpu'), workflow, False)

    def test_manual_and_cached_auto_batches_for_every_workflow(self):
        for workflow in ('semantic', 'segmentation', 'restoration'):
            model = torch.nn.Identity()
            cfg = {'DATA': {'PATCH_SIZE': [2, 3, 4, 1]},
                   'TRAIN': {'BATCH_SIZE': 3}, 'INFER': {'BATCH_SIZE': 2}}
            self.assertEqual(resolve_inference_batch(model, cfg, torch.device('cpu'), workflow), 2)
            cfg['INFER']['AUTO_BATCH_SIZE'] = True
            with patch('sknext.utils.auto_batch.determine_inference_batch_size', return_value=7) as probe:
                for _ in range(2):
                    self.assertEqual(resolve_inference_batch(model, cfg, torch.device('cuda'), workflow), 7)
                probe.assert_called_once()
                self.assertEqual(probe.call_args.args[1], (1, 2, 3, 4))
                self.assertEqual(probe.call_args.kwargs['workflow'], workflow)
            self.assertEqual(cfg['TRAIN']['BATCH_SIZE'], 3)

    def test_search_checks_all_devices_and_recovers_oom(self):
        model = torch.nn.Identity()
        model.device_ids = [0, 1]
        peak = [0]
        def trial(model, shape, batch, device, workflow, amp):
            self.assertFalse(model.training)
            if batch > 12:
                raise torch.cuda.OutOfMemoryError('test')
            peak[0] = batch // 2 * 100
        fork = torch.random.fork_rng
        with ExitStack() as stack:
            for name in ('empty_cache', 'reset_peak_memory_stats', 'synchronize'):
                stack.enter_context(patch('torch.cuda.' + name))
            stack.enter_context(patch('torch.cuda.memory_allocated', return_value=0))
            stack.enter_context(patch('torch.cuda.mem_get_info',
                                     side_effect=lambda gpu: (1000 if gpu == 0 else 500, 1000)))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', side_effect=lambda gpu: peak[0]))
            stack.enter_context(patch('torch.random.fork_rng', side_effect=lambda devices: fork(devices=[])))
            stack.enter_context(patch('sknext.utils.auto_batch._inference_trial', side_effect=trial))
            batch = determine_inference_batch_size(model, (1, 2, 3, 4), torch.device('cuda', 0),
                                                  workflow='restoration', maximum=16, memory_fraction=0.8)
        self.assertEqual(batch, 8)
        self.assertTrue(model.training)

    def test_cpu_auto_request_is_actionable(self):
        with self.assertRaisesRegex(ValueError, 'INFER.AUTO_BATCH_SIZE requires CUDA'):
            determine_inference_batch_size(torch.nn.Identity(), (1, 2, 3, 4),
                                           torch.device('cpu'), workflow='semantic')


if __name__ == '__main__':
    unittest.main()
