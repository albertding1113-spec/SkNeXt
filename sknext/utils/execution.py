"""Single-node process execution, CPU budgets and distributed training utilities.

Imports of torch stay lazy so the CLI can set CUDA visibility before discovery.
"""
import os
from pathlib import Path

_native_thread_limits = None


def _limit_native_threads(threads):
    global _native_thread_limits
    try:
        from threadpoolctl import threadpool_limits
        _native_thread_limits = threadpool_limits(limits=threads)
    except (ImportError, OSError):
        # Optional introspection can fail for partially installed BLAS runtimes.
        # Environment limits and torch's own thread limit still apply.
        pass
    try:
        import cv2
        cv2.setNumThreads(threads)
    except ImportError:
        pass


def world_size():
    return int(os.environ.get('SKNEXT_WORLD_SIZE', '1'))


def rank():
    return int(os.environ.get('SKNEXT_RANK', '0'))


def primary():
    return rank() == 0 and not os.environ.get('SKNEXT_INFER_WORKER')


def cfgvalue(cfg, path, default):
    for key in path.split('.'):
        if key not in cfg:
            return default
        cfg = cfg[key]
    return cfg


def cpu_budget(processes=1):
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', available))
    return max(1, min(available, allocated) // max(1, processes))


def configure_threads(cfg, processes=1):
    import torch
    budget = cpu_budget(processes)
    requested = int(cfgvalue(cfg, 'SYSTEM.CPU_THREADS', 0))
    threads = min(budget, requested) if requested > 0 else min(4, budget)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(threads)
    torch.set_num_threads(threads)
    _limit_native_threads(threads)
    # Compression and Zarr have their own pools; keep them bounded per process.
    try:
        import numcodecs
        numcodecs.blosc.set_nthreads(1)
        import zarr
        zarr.config.set({'async.concurrency': min(8, budget), 'threading.max_workers': min(8, budget)})
    except ImportError:
        pass
    return budget


def local_batch_size(total, size=None):
    size = world_size() if size is None else size
    total = int(total)
    if total < size or total % size:
        raise ValueError(f'TRAIN.BATCH_SIZE={total} must be a positive multiple of {size} GPUs for DDP')
    return total // size


def prepare_once(fn, *args, **kwargs):
    """Run shared data preparation only on rank zero, then broadcast its result."""
    result = fn(*args, **kwargs) if primary() else None
    if world_size() > 1:
        import torch.distributed as dist
        values = [result]
        dist.broadcast_object_list(values, src=0)
        result = values[0]
    return result


def reduce_sums(values, device):
    if world_size() == 1:
        return list(values)
    import torch
    import torch.distributed as dist
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor)
    return tensor.tolist()


def validation_model(model):
    # Validation shards may have different lengths. Avoid DDP forward collectives.
    return model.module if world_size() > 1 else model


def reduce_metrics(loss_sum, count, metrics, device):
    keys = sorted(metrics)
    if world_size() > 1:
        import torch.distributed as dist
        all_keys = [None] * world_size()
        dist.all_gather_object(all_keys, keys)
        keys = sorted(set().union(*all_keys))
    values = reduce_sums([loss_sum, count] + [metrics.get(k, 0.) for k in keys], device)
    return values[0], values[1], dict(zip(keys, values[2:]))


def set_epoch(loader, epoch):
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)


def _loader_worker(worker_id):
    import torch
    import random
    import numpy as np
    torch.set_num_threads(1)
    _limit_native_threads(1)
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % (2 ** 32))
    import numcodecs
    numcodecs.blosc.set_nthreads(1)
    import zarr
    zarr.config.set({'async.concurrency': 2, 'threading.max_workers': 2})


def make_loader(dataset, cfg, device, training=True, shuffle=None):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    batch = local_batch_size(cfgvalue(cfg, 'TRAIN.BATCH_SIZE', 1))
    if len(dataset) == 0:
        raise ValueError('Training and validation patch datasets must not be empty')
    budget = cpu_budget(world_size())
    workers = int(cfgvalue(cfg, 'TRAIN.NUM_WORKERS', -1))
    if workers < 0:
        workers = min(8, max(0, (budget - min(4, budget)) // 2))
    # Training and validation retain separate pools, so budget for both.
    workers = min(workers, max(0, (budget - min(4, budget)) // 2))
    shuffle = training if shuffle is None else shuffle
    sampler = None
    if world_size() > 1:
        if training:
            sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=shuffle)
        else:
            # No duplicate padding in validation: every sample counts exactly once.
            dataset = Subset(dataset, range(rank(), len(dataset), world_size()))
    options = {}
    if workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True,
                       prefetch_factor=int(cfgvalue(cfg, 'TRAIN.PREFETCH_FACTOR', 1)),
                       worker_init_fn=_loader_worker)
    generator = torch.Generator().manual_seed(42 + rank())
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=workers, pin_memory=device.type == 'cuda',
                      generator=generator, **options)


class NullWriter:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def training_rng_states():
    """Collect CPU and local-GPU RNG states without opening other GPUs' contexts."""
    import random
    import numpy as np
    import torch
    rng = np.random.get_state()
    state = dict(rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
                 rng_numpy=(rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]),
                 rng_cuda=[torch.cuda.get_rng_state()] if torch.cuda.is_available() else [],
                 rng_cuda_local=True)
    states = [state]
    if world_size() > 1:
        import torch.distributed as dist
        states = [None] * world_size()
        dist.all_gather_object(states, state)
    return states


def _training_worker(local_rank, size, cfg, run_id, workflow, rendezvous):
    import torch
    import torch.distributed as dist
    from sknext.utils.utils import set_seed
    os.environ['SKNEXT_WORLD_SIZE'] = str(size)
    os.environ['SKNEXT_RANK'] = str(local_rank)
    configure_threads(cfg, size)
    torch.cuda.set_device(local_rank)
    backend = 'nccl' if dist.is_nccl_available() else 'gloo'
    from datetime import timedelta
    timeout = int(cfgvalue(cfg, 'SYSTEM.DISTRIBUTED_TIMEOUT_SECONDS', 21600))
    dist.init_process_group(backend, init_method=rendezvous, rank=local_rank, world_size=size,
                            timeout=timedelta(seconds=timeout))
    try:
        set_seed(42 + local_rank)
        from sknext._sknext import SkNeXt
        runner = SkNeXt.__new__(SkNeXt)
        runner.cfg, runner.run_id, runner.workflow = cfg, run_id, workflow
        runner.device = torch.device('cuda', local_rank)
        # Rank zero owns console output; exceptions on every rank remain visible.
        import contextlib
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink) if local_rank else contextlib.nullcontext():
            runner.run()
    finally:
        dist.destroy_process_group()


def launch_training(cfg, run_id, workflow, size):
    import tempfile
    import torch.multiprocessing as mp
    local_batch_size(cfg.TRAIN.BATCH_SIZE, size)
    configure_threads(cfg, size)
    print(f'Distributed training: {size} GPU processes, global batch {cfg.TRAIN.BATCH_SIZE}, '
          f'local batch {int(cfg.TRAIN.BATCH_SIZE)//size}, CPU budget {cpu_budget(size)}/process', flush=True)
    # File rendezvous avoids fixed-port collisions between cluster jobs.
    with tempfile.TemporaryDirectory(prefix='sknext-ddp-') as folder:
        rendezvous = (Path(folder) / 'rendezvous').resolve().as_uri()
        mp.spawn(_training_worker, args=(size, cfg, run_id, workflow, rendezvous), nprocs=size, join=True)
