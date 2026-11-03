"""Bounded CPU patch preparation, safe to import in spawned workers."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os


def resolve_conversion_workers(workers):
    """Zero selects at most eight available CPUs; one disables multiprocessing."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("DATA.CONVERSION_WORKERS must be a nonnegative integer")
    if workers:
        return workers
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    if hasattr(os, "sched_getaffinity"):
        available = min(available, len(os.sched_getaffinity(0)))
    return min(8, available)


def _initialize_worker():
    # Avoid each patch process launching its own full native CPU thread pool.
    import torch
    torch.set_num_threads(1)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    global _thread_limits
    _thread_limits = threadpool_limits(limits=1)


@contextmanager
def conversion_pool(workers):
    if workers == 1:
        yield None
    else:
        # Never fork a training process with an initialized CUDA runtime.
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_worker) as pool:
            yield pool


def bounded_map(pool, function, jobs, workers):
    """Yield results in source order with at most two jobs per worker in flight."""
    if pool is None:
        yield from map(function, jobs)
        return
    jobs = iter(jobs)
    pending = deque()
    try:
        for _ in range(2 * workers):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(pool.submit(function, job))
        while pending:
            yield pending.popleft().result()
            job = next(jobs, None)
            if job is not None:
                pending.append(pool.submit(function, job))
    finally:
        for future in pending:
            future.cancel()


def prepare_patch(job):
    """Prepare only cropped arrays; source volumes never travel to workers."""
    from sknext.data.preprocessing import preprocess_img, reflect_padding_img
    from sknext.data.label import generate_channels_from_labels

    raw, labels, shape, preprocess, normalize, channels, options, first_labels = job
    raw = reflect_padding_img(raw, shape)
    if normalize:
        raw = preprocess_img(raw, preprocess)
    generated = first_labels if first_labels is not None else generate_channels_from_labels(
        labels, channels, options)
    return raw, [reflect_padding_img(label, shape) for label in generated]
