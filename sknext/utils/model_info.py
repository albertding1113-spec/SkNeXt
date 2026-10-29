# sknext/utils/model_info.py

import torch
import torch.nn as nn
from sknext.utils.progress import log_message


def bytes_to_mb(num_bytes: int) -> float:
    """Convert num_bytes to mebibytes using 1024 squared bytes per unit."""
    return num_bytes / 1024 / 1024


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Compatible with DataParallel / DistributedDataParallel。
    """
    return model.module if hasattr(model, "module") else model


def print_model_parameters(model: nn.Module, print_detail: bool = True):
    """Print parameter counts and storage sizes for a PyTorch model.

    Args:
        model: Module, optionally wrapped in a parallel-training container.
        print_detail: Include each parameter's shape, dtype, device, and size.

    Also reports trainable/frozen totals and buffer storage. Sizes use MiB
    although the console labels them MB; nothing is returned.
    """
    model = unwrap_model(model)

    total_params = 0
    trainable_params = 0
    param_bytes = 0

    log_message("=" * 120)
    log_message("Model parameter summary")
    log_message("=" * 120)

    if print_detail:
        header = (
            f"{'name':70s} "
            f"{'shape':25s} "
            f"{'dtype':15s} "
            f"{'trainable':10s} "
            f"{'numel':15s} "
            f"{'size(MB)':10s} "
            f"{'device':10s}"
        )
        log_message(header)
        log_message("-" * 120)

    for name, param in model.named_parameters():
        numel = param.numel()
        size_bytes = numel * param.element_size()

        total_params += numel
        param_bytes += size_bytes

        if param.requires_grad:
            trainable_params += numel

        if print_detail:
            log_message(
                f"{name:70s} "
                f"{str(tuple(param.shape)):25s} "
                f"{str(param.dtype):15s} "
                f"{str(param.requires_grad):10s} "
                f"{numel:<15,d} "
                f"{bytes_to_mb(size_bytes):<10.4f} "
                f"{str(param.device):10s}"
            )

    buffer_params = 0
    buffer_bytes = 0

    for name, buffer in model.named_buffers():
        numel = buffer.numel()
        size_bytes = numel * buffer.element_size()
        buffer_params += numel
        buffer_bytes += size_bytes

    log_message("-" * 120)
    log_message(f"Total parameters:      {total_params:,}")
    log_message(f"Trainable parameters:  {trainable_params:,}")
    log_message(f"Frozen parameters:     {total_params - trainable_params:,}")
    log_message(f"Parameter memory:      {bytes_to_mb(param_bytes):.2f} MB")
    log_message(f"Buffer elements:       {buffer_params:,}")
    log_message(f"Buffer memory:         {bytes_to_mb(buffer_bytes):.2f} MB")
    log_message(f"Param + buffer memory: {bytes_to_mb(param_bytes + buffer_bytes):.2f} MB")
    log_message("=" * 120)


def print_cuda_memory(device: torch.device | None = None, prefix: str = "", *,
                      all_devices: bool = False):
    """
    Print allocator usage and device-wide free/used memory in MiB.

    Allocated/reserved values belong to this PyTorch process. Peak values are
    measured since the last CUDA peak reset. Device used includes other processes.
    Explicit CPU devices skip CUDA queries, even on machines with a GPU.
    With all_devices=True, report every CUDA device visible to this process.
    """
    if device is not None and device.type != "cuda":
        return
    if not torch.cuda.is_available():
        log_message("CUDA is not available.", flush=True)
        return

    # Process workers own one device. Querying every device here would create
    # unnecessary CUDA contexts on their peers (and report the wrong allocator).
    from sknext.utils.execution import world_size
    if all_devices and world_size() > 1:
        all_devices = False
    if all_devices:
        for index in range(torch.cuda.device_count()):
            print_cuda_memory(torch.device(f"cuda:{index}"), prefix=prefix)
        return

    if device is None:
        device = torch.device("cuda:0")

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    log_message("=" * 80, flush=True)
    log_message(f"CUDA memory summary {prefix}")
    log_message("=" * 80, flush=True)
    log_message(f"Device:                {device}")
    log_message(f"Allocated:             {bytes_to_mb(allocated):.2f} MiB")
    log_message(f"Reserved:              {bytes_to_mb(reserved):.2f} MiB")
    log_message(f"Max allocated:         {bytes_to_mb(max_allocated):.2f} MiB")
    log_message(f"Max reserved:          {bytes_to_mb(max_reserved):.2f} MiB")
    log_message(f"Device used:           {bytes_to_mb(total_bytes - free_bytes):.2f} MiB")
    log_message(f"Free memory:           {bytes_to_mb(free_bytes):.2f} MiB")
    log_message(f"Total memory:          {bytes_to_mb(total_bytes):.2f} MiB")
    log_message("=" * 80, flush=True)


@torch.no_grad()
def profile_forward_memory(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    """
    Use dummy input to test max forward memory.

    input_shape:
        3D segmentation: (B, C, Z, Y, X)
        2D segmentation: (B, C, Y, X)
    """
    model = unwrap_model(model)
    model.to(device)
    model.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    x = torch.randn(input_shape, device=device, dtype=dtype)

    print_cuda_memory(device, prefix="before forward")

    out = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print_cuda_memory(device, prefix="after forward")

    if isinstance(out, dict):
        log_message("Output:")
        for key, value in out.items():
            log_message(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    else:
        log_message(f"Output shape: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")

    return out
