import math
from collections.abc import Sequence

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


def _resolve_total_steps(
    total_steps: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
) -> int:
    if total_steps is not None:
        assert total_steps > 0, "total_steps must be > 0."
        return int(total_steps)
    assert epochs is not None and steps_per_epoch is not None, (
        "Please provide either total_steps or both epochs and steps_per_epoch."
    )
    assert epochs > 0, "epochs must be > 0."
    assert steps_per_epoch > 0, "steps_per_epoch must be > 0."
    return int(epochs * steps_per_epoch)


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    warmup_steps: int | None = None,
    warmup_epochs: int | None = None,
    min_lr: float = 1e-6,
    warmup_start_factor: float = 0.0,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Build a PyTorch LambdaLR scheduler with linear warmup + cosine decay.

    Each iteration:
        optimizer.step()
        scheduler.step()
    """
    total_steps = _resolve_total_steps(total_steps, epochs, steps_per_epoch)

    if warmup_steps is None:
        if warmup_epochs is None:
            warmup_steps = 0
        else:
            assert steps_per_epoch is not None, (
                "steps_per_epoch is required when warmup_epochs is provided."
            )
            warmup_steps = int(warmup_epochs * steps_per_epoch)

    warmup_steps = int(warmup_steps)

    assert warmup_steps >= 0, "warmup_steps must be >= 0."
    assert warmup_steps < total_steps, "warmup_steps must be smaller than total_steps."
    assert min_lr >= 0, "min_lr must be >= 0."
    assert 0.0 <= warmup_start_factor <= 1.0, "warmup_start_factor must be in [0, 1]."

    base_lrs = [group["lr"] for group in optimizer.param_groups]
    assert all(lr > 0 for lr in base_lrs), "All optimizer learning rates must be > 0."

    min_lr_factors = [min_lr / base_lr for base_lr in base_lrs]

    def make_lr_lambda(min_lr_factor: float):
        def lr_lambda(current_step: int) -> float:
            current_step = min(current_step, total_steps)

            # 1. Linear warmup
            if warmup_steps > 0 and current_step < warmup_steps:
                alpha = current_step / warmup_steps
                return warmup_start_factor + alpha * (1.0 - warmup_start_factor)

            # 2. Cosine decay
            decay_steps = total_steps - warmup_steps
            decay_step = current_step - warmup_steps
            progress = decay_step / max(1, decay_steps)
            progress = min(max(progress, 0.0), 1.0)

            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            return min_lr_factor + (1.0 - min_lr_factor) * cosine_factor

        return lr_lambda

    lr_lambdas = [
        make_lr_lambda(min_lr_factor)
        for min_lr_factor in min_lr_factors
    ]
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambdas,
        last_epoch=last_epoch,
    )


def build_onecycle_scheduler(
    optimizer: Optimizer,
    *,
    max_lr: float | Sequence[float],
    total_steps: int | None = None,
    epochs: int | None = None,
    steps_per_epoch: int | None = None,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    three_phase: bool = False,
    last_epoch: int = -1,
) -> OneCycleLR:
    """
    Build a PyTorch OneCycleLR scheduler.

    OneCycleLR 必须按 iteration 调用，不能按 epoch 调用：
        optimizer.step()
        scheduler.step()

    Parameters
    ----------
    optimizer:
        PyTorch optimizer.
    max_lr:
        Peak learning rate. For multiple parameter groups, can be a list.
    total_steps:
        Total scheduler steps. Usually epochs * steps_per_epoch.
    epochs:
        Total training epochs. Used only when total_steps is None.
    steps_per_epoch:
        Number of batches per epoch. Used only when total_steps is None.
    pct_start:
        Percentage of cycle spent increasing lr.
    anneal_strategy:
        "cos" or "linear".
    div_factor:
        initial_lr = max_lr / div_factor.
    final_div_factor:
        min_lr = initial_lr / final_div_factor.
    three_phase:
        Whether to use three-phase OneCycle schedule.
    last_epoch:
        PyTorch scheduler last_epoch argument.

    Returns
    -------
    OneCycleLR
        PyTorch learning rate scheduler.
    """
    total_steps = _resolve_total_steps(total_steps, epochs, steps_per_epoch)

    assert 0.0 < pct_start < 1.0, "pct_start must be in (0, 1)."
    assert anneal_strategy in ["cos", "linear"], "anneal_strategy must be 'cos' or 'linear'."
    assert div_factor > 0, "div_factor must be > 0."
    assert final_div_factor > 0, "final_div_factor must be > 0."

    return OneCycleLR(
        optimizer=optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
        last_epoch=last_epoch,
    )

