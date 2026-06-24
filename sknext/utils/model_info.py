# sknext/utils/model_info.py

import torch
import torch.nn as nn


def bytes_to_mb(num_bytes: int) -> float:
    return num_bytes / 1024 / 1024


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Compatible with DataParallel / DistributedDataParallel。
    """
    return model.module if hasattr(model, "module") else model


def print_model_parameters(model: nn.Module, print_detail: bool = True):
    model = unwrap_model(model)

    total_params = 0
    trainable_params = 0
    param_bytes = 0

    print("=" * 120)
    print("Model parameter summary")
    print("=" * 120)

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
        print(header)
        print("-" * 120)

    for name, param in model.named_parameters():
        numel = param.numel()
        size_bytes = numel * param.element_size()

        total_params += numel
        param_bytes += size_bytes

        if param.requires_grad:
            trainable_params += numel

        if print_detail:
            print(
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

    print("-" * 120)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {total_params - trainable_params:,}")
    print(f"Parameter memory:      {bytes_to_mb(param_bytes):.2f} MB")
    print(f"Buffer elements:       {buffer_params:,}")
    print(f"Buffer memory:         {bytes_to_mb(buffer_bytes):.2f} MB")
    print(f"Param + buffer memory: {bytes_to_mb(param_bytes + buffer_bytes):.2f} MB")
    print("=" * 120)


def print_cuda_memory(device: torch.device | None = None, prefix: str = ""):
    """
    Print current cuda memory usage
    Notice:
    - memory_allocated: currently used memory
    - memory_reserved: reserved memory
    - max_memory_allocated: max allocated memory
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if device is None:
        device = torch.device("cuda:0")

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    print("=" * 80)
    print(f"CUDA memory summary {prefix}")
    print("=" * 80)
    print(f"Device:                {device}")
    print(f"Allocated:             {bytes_to_mb(allocated):.2f} MB")
    print(f"Reserved:              {bytes_to_mb(reserved):.2f} MB")
    print(f"Max allocated:         {bytes_to_mb(max_allocated):.2f} MB")
    print(f"Max reserved:          {bytes_to_mb(max_reserved):.2f} MB")
    print(f"Free memory:           {bytes_to_mb(free_bytes):.2f} MB")
    print(f"Total memory:          {bytes_to_mb(total_bytes):.2f} MB")
    print("=" * 80)


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
        print("Output:")
        for key, value in out.items():
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    else:
        print(f"Output shape: {tuple(out.shape)}, dtype={out.dtype}, device={out.device}")

    return out