"""CUDA selection and shared batch parallel execution for every workflow."""
import os
import re
import warnings


def configure_visible_devices(selector):
    """Set CUDA visibility before CUDA initialization; 'all' honors external masks."""
    selector = str(selector).strip()
    if selector.lower() == "all":
        return
    ids = [item.strip() for item in selector.split(",")]
    if (not ids or any(not re.fullmatch(r"(?:\d+|GPU-[\w-]+|MIG-[\w/-]+)", item)
                       for item in ids) or len(set(ids)) != len(ids)):
        raise ValueError("--gpu must be distinct GPU indices/UUIDs separated by commas, or 'all'")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids)


def parallel_model(model, device, batch_size=None):
    """Split global batches over visible GPUs, gathering outputs on the primary."""
    import torch
    model = model.to(device)
    if device.type != "cuda":
        return model
    from sknext.utils.execution import world_size
    if world_size() > 1:
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index,
            broadcast_buffers=False, gradient_as_bucket_view=True)
    if os.environ.get('SKNEXT_INFER_WORKER'):
        return model
    count = torch.cuda.device_count()
    if count <= 1:
        return model
    primary = device.index if device.index is not None else torch.cuda.current_device()
    ids = [primary] + [i for i in range(count) if i != primary]
    if batch_size is not None and int(batch_size) < count:
        warnings.warn(f"TRAIN.BATCH_SIZE={batch_size} is smaller than the {count} selected GPUs; "
                      "increase it to use every GPU in each full training batch.", stacklevel=2)
    print(f"Batch parallel execution on CUDA devices {ids}", flush=True)
    return torch.nn.DataParallel(model, device_ids=ids, output_device=primary)


def inference_batch_size(model, configured):
    """Allow at least one inference patch per selected GPU."""
    return max(1, int(configured), len(getattr(model, "device_ids", [])))


def model_state_dict(model):
    """Keep saved weights portable between single and multi-GPU runs."""
    return (model.module if hasattr(model, "module") else model).state_dict()


def configured_inference_batch(cfg):
    from sknext.utils.execution import cfgvalue
    value = int(cfgvalue(cfg, 'INFER.BATCH_SIZE', 0))
    return value if value > 0 else int(cfgvalue(cfg, 'TRAIN.BATCH_SIZE', 1))


def reduce_inference_batch(model, failed_batch, device):
    """Release failed CUDA allocations and cache a 20% smaller retry batch.

    Call after leaving the exception handler so its traceback is released.
    """
    import gc
    import torch
    from sknext.utils.progress import log_message
    batch = max(1, int(failed_batch * 0.8))
    gc.collect()
    torch.cuda.empty_cache()
    cached = getattr(model, '_sknext_inference_batch', None)
    if cached is not None:
        model._sknext_inference_batch = (cached[0], batch)
    log_message(f'[AUTO INFER BATCH] {device}: CUDA OOM; retrying with batch={batch}',
                flush=True)
    return batch


def predict_inference_batches(model, cfg, device, inputs, predict):
    """Retry pure predictions on OOM, returning CPU arrays in input order.

    The callback owns GPU tensors and must return a NumPy array before any
    blending or output writes. Failed attempts therefore have no side effects.
    """
    import numpy as np
    import torch
    from sknext.utils.execution import cfgvalue
    automatic = bool(cfgvalue(cfg, 'INFER.AUTO_BATCH_SIZE', False))
    cached = getattr(model, '_sknext_inference_batch', None)
    batch = min(len(inputs), cached[1]) if automatic and cached else len(inputs)
    outputs = []
    start = 0
    while start < len(inputs):
        end = min(start + batch, len(inputs))
        retry = False
        try:
            result = predict(inputs[start:end])
        except torch.cuda.OutOfMemoryError:
            if not automatic or end - start <= 1:
                raise
            retry = True
        if retry:
            batch = reduce_inference_batch(model, end - start, device)
            continue
        outputs.append(result)
        start = end
    return outputs[0] if len(outputs) == 1 else np.concatenate(outputs, axis=0)


def resolve_inference_batch(model, cfg, device, workflow, *, amp=False):
    """Reuse a per-model probe result, or preserve manual inference batching."""
    from sknext.utils.execution import cfgvalue
    if not bool(cfgvalue(cfg, 'INFER.AUTO_BATCH_SIZE', False)):
        return inference_batch_size(model, configured_inference_batch(cfg))
    from sknext.utils.auto_batch import determine_inference_batch_size
    patch = tuple(int(v) for v in cfgvalue(cfg, 'DATA.PATCH_SIZE', ()))
    if len(patch) != 4 or min(patch) < 1:
        raise ValueError('Automatic inference sizing requires positive DATA.PATCH_SIZE in ZYXC order')
    maximum = int(cfgvalue(cfg, 'INFER.AUTO_BATCH_MAX_PER_GPU', 64))
    fraction = float(cfgvalue(cfg, 'INFER.AUTO_BATCH_MEMORY_FRACTION', 0.85))
    key = (patch, str(device), workflow, amp, maximum, fraction)
    cached = getattr(model, '_sknext_inference_batch', None)
    if cached is None or cached[0] != key:
        batch = determine_inference_batch_size(
            model, (patch[-1], *patch[:3]), device, workflow=workflow,
            amp=amp, maximum=maximum, memory_fraction=fraction)
        model._sknext_inference_batch = (key, batch)
    return model._sknext_inference_batch[1]
