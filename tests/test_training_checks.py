"""Numerical failures must stop before invalid optimizer updates."""
import unittest
from unittest.mock import patch

import torch

from sknext.utils.training_checks import check_finite


class TrainingChecksTests(unittest.TestCase):
    def check(self, tensors, **kwargs):
        check_finite(tensors, context='[TRAIN] [epoch=63] [batch=1700]',
                     device=torch.device('cpu'), **kwargs)

    def test_finite_values_and_unused_gradients(self):
        self.check([('input', torch.tensor([0., -1., 1.])), ('unused', None)])
        self.check([])

    def test_nan_and_both_infinities_report_tensor_and_location(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaisesRegex(FloatingPointError,
                                            r'epoch=63.*batch=1700.*logits.*nonfinite=1'):
                    self.check([('logits', torch.tensor([1., value]))])

    def test_invalid_gradient_prevents_parameter_update(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        model(torch.ones(1, 1)).sum().backward()
        model.weight.grad.fill_(float('nan'))
        with self.assertRaisesRegex(FloatingPointError, 'gradient:weight'):
            self.check((('gradient:' + name, p.grad) for name, p in model.named_parameters()))
            optimizer.step()
        torch.testing.assert_close(model.weight, before)

    def test_finite_gradients_allow_update(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        before = model.weight.detach().clone()
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        scaler.scale(model(torch.ones(1, 1)).sum()).backward()
        scaler.unscale_(optimizer)
        self.check((('gradient:' + name, p.grad) for name, p in model.named_parameters()))
        scaler.step(optimizer)
        scaler.update()
        torch.testing.assert_close(model.weight, before - 0.1)

    def test_remote_failure_stops_healthy_rank(self):
        with patch('sknext.utils.training_checks.dist.is_initialized', return_value=True), \
             patch('sknext.utils.training_checks.dist.get_rank', return_value=1), \
             patch('sknext.utils.training_checks.dist.all_reduce',
                   side_effect=lambda flag, op: flag.zero_()):
            with self.assertRaisesRegex(FloatingPointError, 'another rank'):
                self.check([('inputs', torch.ones(2))], synchronize=True)

    def test_validation_does_not_use_collectives(self):
        with patch('sknext.utils.training_checks.dist.all_reduce') as reduce:
            self.check([('inputs', torch.ones(2))])
            reduce.assert_not_called()


if __name__ == '__main__':
    unittest.main()
