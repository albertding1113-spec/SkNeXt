"""Startup CUDA memory probing with disposable models and optimizer state."""
import gc
import math
import random

import numpy as np
import torch
import torch.distributed as dist

from sknext.utils.progress import log_message


def _training_trial(model_factory, loss_fn, sample_x, sample_y, batch, device,
                    amp, optimizer_options, optimizer_name='ADAMW'):
    model = model_factory().to(device).train()
    optimizer_cls = {'ADAMW': torch.optim.AdamW, 'ADAM': torch.optim.Adam,
                     'SGD': torch.optim.SGD}[optimizer_name.upper()]
    optimizer = optimizer_cls(model.parameters(), **optimizer_options)
    x = sample_x.to(device).unsqueeze(0).repeat(batch, *([1] * sample_x.ndim))
    y = sample_y.to(device).unsqueeze(0).repeat(batch, *([1] * sample_y.ndim))
    # Include allocated Adam moments and temporary optimizer buffers in the peak.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            prediction = model(x)
            if isinstance(prediction, dict):
                prediction = prediction['pred']
            loss = loss_fn(prediction.float(), y)
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def largest_fitting_batch(probe, maximum):
    """Exponential search followed by binary search; zero means batch one failed."""
    if maximum < 1:
        raise ValueError('Automatic batch maximum must be positive')
    low, candidate = 0, 1
    while True:
        if not probe(candidate):
            high = candidate - 1
            break
        low = candidate
        if candidate == maximum:
            return low
        candidate = min(maximum, candidate * 2)
    while low < high:
        middle = (low + high + 1) // 2
        if probe(middle):
            low = middle
        else:
            high = middle - 1
    return low


def determine_batch_size(model_factory, loss_fn, dataset, device, *, amp,
                         optimizer_options, maximum=64, memory_fraction=0.85,
                         optimizer_name='ADAMW'):
    """Return a global batch after two full optimizer steps per candidate.

    The factory must return an unwrapped model. Probes use repeated copies of
    one real patch and discard all weights/state afterwards. Memory headroom
    allows for DDP buckets and runtime variation; this is not an OOM guarantee.
    """
    if device.type != 'cuda':
        raise ValueError('TRAIN.AUTO_BATCH_SIZE requires CUDA')
    if not 0 < memory_fraction < 1:
        raise ValueError('TRAIN.AUTO_BATCH_MEMORY_FRACTION must be between 0 and 1')
    if maximum < 1 or len(dataset) < 1:
        raise ValueError('Automatic batch sizing requires positive maximum and nonempty data')
    distributed = dist.is_available() and dist.is_initialized()
    size = dist.get_world_size() if distributed else 1
    if size == 1 and torch.cuda.device_count() > 1:
        raise ValueError('Automatic batch sizing on multiple GPUs requires the main sknext CLI (DDP)')
    maximum = min(maximum, math.ceil(len(dataset) / size))
    index = device.index if device.index is not None else torch.cuda.current_device()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[index]):
            sample_x, sample_y = dataset[0]
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(device)
            budget = int(free * memory_fraction)

            def probe(batch):
                log_message(f'[AUTO BATCH] GPU {index}: probing batch={batch}', flush=True)
                torch.cuda.empty_cache()
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                fits = False
                try:
                    _training_trial(model_factory, loss_fn, sample_x, sample_y,
                                    batch, device, amp, optimizer_options,
                                    **({'optimizer_name': optimizer_name} if optimizer_name != 'ADAMW' else {}))
                    fits = torch.cuda.max_memory_allocated(device) - baseline <= budget
                except torch.cuda.OutOfMemoryError:
                    pass
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
                return fits

            local = largest_fitting_batch(probe, maximum)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if distributed:
        value = torch.tensor(local, device=device, dtype=torch.int64)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        local = int(value)
    if local == 0:
        raise RuntimeError('Automatic batch sizing: even one patch exceeds the GPU memory budget. '
                           'Reduce DATA.PATCH_SIZE or model size, or free GPU memory.')
    return local * size


def configure_auto_batch(cfg, model_factory, loss_fn, dataset, device, latest_path):
    """Select or restore the batch before building loaders and schedulers."""
    from sknext.utils.checkpoint import read_checkpoint
    if cfg.TRAIN.RESUME:
        batch = read_checkpoint(latest_path).get('batch_size')
        if batch is None:
            raise ValueError('Checkpoint has no saved batch size. Disable TRAIN.AUTO_BATCH_SIZE '
                             'and set TRAIN.BATCH_SIZE to the original batch size to resume.')
        batch = int(batch)
        if batch < 1:
            raise ValueError('Checkpoint batch size must be positive')
    else:
        name = str(cfg.TRAIN.OPTIMIZER).upper()
        if name not in {'SGD', 'ADAM', 'ADAMW'}:
            raise ValueError(f'Unsupported optimizer {name}')
        options = dict(lr=float(cfg.TRAIN.LR), weight_decay=float(cfg.TRAIN.W_DECAY))
        options.update(dict(momentum=0.9) if name == 'SGD' else
                       dict(betas=tuple(cfg.TRAIN.OPT_BETAS)))
        batch = determine_batch_size(
            model_factory, loss_fn, dataset, device, amp=bool(cfg.TRAIN.AMP),
            optimizer_options=options, optimizer_name=name,
            maximum=int(cfg.TRAIN.AUTO_BATCH_MAX_PER_GPU),
            memory_fraction=float(cfg.TRAIN.AUTO_BATCH_MEMORY_FRACTION))
    cfg.TRAIN.BATCH_SIZE = batch
    size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    log_message(f'Automatic batch size: global={batch}, per GPU={batch // size}', flush=True)
    return batch


def _inference_trial(model, shape, batch, device, workflow, amp):
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        inputs = torch.zeros((batch, *shape), device=device, dtype=torch.float32)
        prediction = model(inputs)
        if isinstance(prediction, dict):
            prediction = prediction['pred']
        if workflow == 'semantic':
            prediction = torch.sigmoid(prediction.float())
        elif workflow == 'segmentation':
            prediction = torch.sigmoid(prediction)
        else:
            prediction = prediction.float()
        prediction.cpu()


def determine_inference_batch_size(model, shape, device, *, workflow, amp=False,
                                   maximum=64, memory_fraction=0.85):
    """Probe loaded weights without gradients; return the batch for this process.

    Block workers tune independently. DataParallel fallback checks memory on
    every participating GPU and searches equal per-GPU batch allocations.
    """
    if device.type != 'cuda':
        raise ValueError('INFER.AUTO_BATCH_SIZE requires CUDA')
    if maximum < 1 or not 0 < memory_fraction < 1:
        raise ValueError('INFER automatic batch maximum must be positive and memory fraction between 0 and 1')
    index = device.index if device.index is not None else torch.cuda.current_device()
    devices = list(getattr(model, 'device_ids', [])) or [index]
    count = len(devices)
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.cuda.empty_cache()
            budgets = {gpu: int(torch.cuda.mem_get_info(gpu)[0] * memory_fraction) for gpu in devices}

            def probe(local):
                log_message(f'[AUTO INFER BATCH] GPU {devices}: probing batch={local * count}', flush=True)
                torch.cuda.empty_cache()
                baseline = {gpu: torch.cuda.memory_allocated(gpu) for gpu in devices}
                for gpu in devices:
                    torch.cuda.reset_peak_memory_stats(gpu)
                try:
                    for _ in range(2):
                        _inference_trial(model, shape, local * count, device, workflow, amp)
                    for gpu in devices:
                        torch.cuda.synchronize(gpu)
                    return all(torch.cuda.max_memory_allocated(gpu) - baseline[gpu] <= budgets[gpu]
                               for gpu in devices)
                except torch.cuda.OutOfMemoryError:
                    return False
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            local = largest_fitting_batch(probe, maximum)
    finally:
        model.train(was_training)
    if local == 0:
        raise RuntimeError('Automatic inference batch sizing: even one patch per GPU exceeds '
                           'the memory budget. Reduce DATA.PATCH_SIZE or free GPU memory.')
    batch = local * count
    log_message(f'Automatic inference batch size: process={batch}, per GPU={local}', flush=True)
    return batch
