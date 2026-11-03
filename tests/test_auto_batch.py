"""Batch search and memory-probe orchestration without requiring CUDA hardware."""
from contextlib import ExitStack
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sknext.utils.auto_batch import (largest_fitting_batch, determine_batch_size,
                                     _training_trial, configure_auto_batch)


class AutoBatchTests(unittest.TestCase):
    def test_configure_uses_selected_optimizer_and_restores_saved_batch(self):
        cfg = SimpleNamespace(TRAIN=SimpleNamespace(
            RESUME=False, OPTIMIZER='SGD', LR=0.1, W_DECAY=0.02,
            OPT_BETAS=(0.9, 0.999), AMP=True, AUTO_BATCH_MAX_PER_GPU=32,
            AUTO_BATCH_MEMORY_FRACTION=0.8, BATCH_SIZE=1))
        with patch('sknext.utils.auto_batch.determine_batch_size', return_value=6) as probe:
            self.assertEqual(configure_auto_batch(cfg, None, None, None,
                                                 torch.device('cpu'), 'latest'), 6)
            self.assertEqual(cfg.TRAIN.BATCH_SIZE, 6)
            self.assertEqual(probe.call_args.kwargs['optimizer_name'], 'SGD')
            self.assertEqual(probe.call_args.kwargs['optimizer_options'],
                             dict(lr=0.1, weight_decay=0.02, momentum=0.9))
        cfg.TRAIN.RESUME = True
        with patch('sknext.utils.auto_batch.determine_batch_size') as probe, \
                patch('sknext.utils.checkpoint.read_checkpoint', return_value={'batch_size': 4}):
            self.assertEqual(configure_auto_batch(cfg, None, None, None,
                                                 torch.device('cpu'), 'latest'), 4)
            probe.assert_not_called()
        with patch('sknext.utils.checkpoint.read_checkpoint', return_value={}):
            with self.assertRaisesRegex(ValueError, 'no saved batch size'):
                configure_auto_batch(cfg, None, None, None, torch.device('cpu'), 'latest')

    def test_trials_support_adam_and_sgd(self):
        for name, options in [('ADAM', {'lr': 0.1}), ('SGD', {'lr': 0.1, 'momentum': 0.9})]:
            with self.subTest(optimizer=name):
                model = torch.nn.Linear(1, 1, bias=False)
                model.weight.data.fill_(1.)
                _training_trial(lambda: model, torch.nn.MSELoss(), torch.ones(1), torch.zeros(1),
                                3, torch.device('cpu'), False, options, name)
                self.assertLess(float(model.weight.detach()), 0.85)

    def test_search_boundaries_and_non_power_of_two_limits(self):
        for maximum in (1, 3, 16, 63, 64):
            for capacity in range(maximum + 2):
                seen = []
                def probe(batch):
                    seen.append(batch)
                    return batch <= capacity
                self.assertEqual(largest_fitting_batch(probe, maximum), min(capacity, maximum))
                self.assertEqual(len(seen), len(set(seen)))
                self.assertTrue(all(1 <= batch <= maximum for batch in seen))

    def test_search_propagates_non_memory_errors(self):
        with self.assertRaisesRegex(RuntimeError, 'bad shape'):
            largest_fitting_batch(lambda batch: (_ for _ in ()).throw(RuntimeError('bad shape')), 8)

    def test_actual_trial_runs_two_optimizer_updates(self):
        models = []
        def factory():
            model = torch.nn.Linear(1, 1, bias=False)
            model.weight.data.fill_(1.)
            models.append(model)
            return model
        _training_trial(factory, torch.nn.MSELoss(), torch.ones(1), torch.zeros(1),
                        3, torch.device('cpu'), False, {'lr': 0.1})
        self.assertLess(float(models[0].weight.detach()), 0.85)

    def run_probe(self, *, capacity=6, fraction=0.85, remote=None, fail=None):
        seen = []
        peak = [0]
        def trial(factory, loss, x, y, batch, device, amp, options):
            seen.append(batch)
            random.random()
            np.random.rand()
            torch.rand(1)
            if fail:
                raise fail
            if batch > capacity:
                raise torch.cuda.OutOfMemoryError('test OOM')
            peak[0] = 100 * batch
        fork = torch.random.fork_rng
        with ExitStack() as stack:
            for name, value in [('empty_cache', None), ('reset_peak_memory_stats', None),
                                ('memory_allocated', 0), ('mem_get_info', (1000, 1000)),
                                ('device_count', 1)]:
                stack.enter_context(patch('torch.cuda.' + name, return_value=value))
            stack.enter_context(patch('torch.cuda.max_memory_allocated', side_effect=lambda d: peak[0]))
            stack.enter_context(patch('torch.random.fork_rng', side_effect=lambda devices: fork(devices=[])))
            stack.enter_context(patch('sknext.utils.auto_batch._training_trial', side_effect=trial))
            stack.enter_context(patch('sknext.utils.auto_batch.dist.is_initialized', return_value=remote is not None))
            stack.enter_context(patch('sknext.utils.auto_batch.dist.get_world_size', return_value=2))
            if remote is not None:
                tensor = torch.tensor
                stack.enter_context(patch('sknext.utils.auto_batch.torch.tensor',
                                          side_effect=lambda data, **kw: tensor(data, dtype=kw['dtype'])))
                stack.enter_context(patch('sknext.utils.auto_batch.dist.all_reduce',
                                          side_effect=lambda value, op: value.fill_(min(int(value), remote))))
            result = determine_batch_size(lambda: None, None,
                [(torch.ones(1), torch.zeros(1))] * 32, torch.device('cuda', 0),
                amp=False, optimizer_options={}, maximum=16, memory_fraction=fraction)
        return result, seen

    def test_recovers_from_probe_oom_and_respects_headroom(self):
        self.assertEqual(self.run_probe(capacity=6)[0], 6)
        self.assertEqual(self.run_probe(capacity=16, fraction=0.5)[0], 5)

    def test_smallest_gpu_capacity_determines_global_batch(self):
        self.assertEqual(self.run_probe(capacity=6, remote=3)[0], 6)

    def test_batch_one_failure_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, 'even one patch'):
            self.run_probe(capacity=0)
        with self.assertRaisesRegex(RuntimeError, 'even one patch'):
            self.run_probe(remote=0)

    def test_rng_restored_on_success_and_error(self):
        for failure in (None, ValueError('bad input')):
            py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
            if failure is None:
                self.run_probe()
            else:
                with self.assertRaisesRegex(ValueError, 'bad input'):
                    self.run_probe(fail=failure)
            self.assertEqual(random.getstate(), py_state)
            after_np = np.random.get_state()
            np.testing.assert_array_equal(after_np[1], np_state[1])
            self.assertEqual(after_np[2:], np_state[2:])
            torch.testing.assert_close(torch.get_rng_state(), th_state)

    def test_cpu_request_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, 'requires CUDA'):
            determine_batch_size(None, None, [None], torch.device('cpu'),
                                 amp=False, optimizer_options={})


if __name__ == '__main__':
    unittest.main()
