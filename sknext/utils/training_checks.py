"""Fail-fast numerical checks for training and validation."""
import torch
import torch.distributed as dist


def check_finite(named_tensors, *, context, device, synchronize=False):
    """Reject NaN/Inf, synchronizing failures across equal-length DDP train loops.

    Do not synchronize per-batch validation checks: validation shards can have
    different lengths. Missing gradients (unused parameters) are allowed.
    """
    tensors = [(name, tensor.detach()) for name, tensor in named_tensors
               if tensor is not None]
    flags = [torch.isfinite(tensor).all() for _, tensor in tensors]
    valid = (torch.stack(flags).all() if flags else
             torch.tensor(True, device=device))
    distributed = synchronize and dist.is_available() and dist.is_initialized()
    if distributed:
        valid = valid.to(dtype=torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if bool(valid):
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    details = []
    for (name, tensor), finite in zip(tensors, flags):
        if not bool(finite):
            details.append(
                f"{name} (shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                f"nonfinite={int((~torch.isfinite(tensor)).sum())})")
    reason = '; '.join(details) or 'non-finite values detected on another rank'
    raise FloatingPointError(
        f"{context} [rank={rank}] {reason}. Stopping before further updates or "
        "checkpoint saves; restart from a known-good checkpoint."
    )
