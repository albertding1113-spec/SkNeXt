"""Persistent, one-GPU block workers; only the coordinator writes output stores."""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from sknext.utils.execution import cfgvalue, configure_threads, cpu_budget

_runtime = None


def parallel_inference_enabled(cfg, device):
    import torch
    return (device.type == 'cuda' and not os.environ.get('SKNEXT_INFER_WORKER')
            and bool(cfgvalue(cfg, 'INFER.PARALLEL_BLOCKS', True))
            and torch.cuda.device_count() > 1)


def _initialize(gpu, size, cfg, run_id, workflow):
    global _runtime
    os.environ['SKNEXT_INFER_WORKER'] = '1'
    import torch
    from pathlib import Path
    configure_threads(cfg, size)
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda', gpu)
    # A bounded per-device patch batch; never multiply a global batch by GPU count.
    batch = int(cfgvalue(cfg, 'INFER.BATCH_SIZE', 0))
    if batch <= 0:
        batch = max(1, (int(cfgvalue(cfg, 'TRAIN.BATCH_SIZE', 1)) + size - 1) // size)
    cfg['TRAIN']['BATCH_SIZE'] = batch
    if workflow == 'semantic':
        from sknext.run.semantic_workflow import make_model, load_model_checkpoint, checked_semantic_names
        runner = None
        model = make_model(cfg, device)
        load_model_checkpoint(model, cfg, run_id, device)
        names = checked_semantic_names(cfg)
    elif workflow == 'restoration':
        from sknext.run.restoration_workflow import RestorationWorkflow
        runner = RestorationWorkflow(cfg, device, run_id)
        model = runner.build_model()
        path = Path(cfg.MODEL.CHECKPOINT_INPUT_PATH) if cfg.MODEL.CHECKPOINT_INPUT_PATH else runner.checkpoint
        runner._load(model, path)
        names = None
    else:
        from sknext.run.networkflow import Segmentation_Workflow
        runner = Segmentation_Workflow(cfg, device, run_id)
        runner.set_model()
        runner.load_checkpoint()
        runner.define_postprocess_channels()
        runner.infer_path = Path(runner.infer_path)
        runner._configure_infer_output()
        runner.validate_watershed_config()
        runner.create_reader()
        runner.raw_all_blocks = bool(cfg.INFER.RAW_ALL_BLOCKS) and runner.save_raw_predictions
        if runner.skeleton_flag and (runner.save_segmentation or not runner.raw_all_blocks):
            runner.create_skeleton()
        model, names = runner.model, None
    model.eval()
    _runtime = dict(cfg=cfg, model=model, runner=runner, names=names, device=device,
                    workflow=workflow, reader=None, source=None)
    print(f'[INFER worker {gpu}] ready: batch={batch}, CPU budget={cpu_budget(size)}', flush=True)


def _predict(task):
    import time
    import torch
    started = time.perf_counter()
    result = _predict_impl(task)
    gpu = torch.cuda.current_device()
    print(f'[INFER GPU {gpu}] block {task["id"]}: {time.perf_counter()-started:.2f}s '
          '(read, predict, postprocess)', flush=True)
    if (task['id'] + 1) % 20 == 0:
        from sknext.utils.model_info import print_cuda_memory
        print_cuda_memory(_runtime['device'], prefix=f'[INFER GPU {gpu}]')
    return result


def _predict_impl(task):
    import numpy as np
    import torch
    from pathlib import Path
    rt = _runtime
    runner, model, cfg = rt['runner'], rt['model'], rt['cfg']
    with torch.inference_mode():
        if rt['workflow'] == 'segmentation':
            runner.inferred_block_num = task['id']
            runner.total_block_num = task['total']
            return runner._infer_one_block(task['coord'], task['skeleton']) if task['predict'] else (None, None)
        source = task['source']
        if source != rt['source']:
            if rt['reader'] is not None:
                rt['reader'].close()
            if rt['workflow'] == 'semantic':
                from sknext.run.semantic_workflow import _input_reader
                rt['reader'], rt['channels'] = _input_reader(Path(source), list(cfg['DATA']['INFER']['CHANNEL']))
            else:
                rt['reader'] = runner._open_reader(Path(source))
            rt['source'] = source
        if rt['workflow'] == 'restoration':
            return runner._predict_sr_block(model, rt['reader'], task['start'], task['stop'],
                                            task['bounds'], task['geometry'])
        from sknext.run.semantic_workflow import _predict_block
        low, high = task['low'], task['high']
        key = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
        block = np.stack([rt['reader'][(ch, *key)] for ch in rt['channels']], axis=0)
        probs = _predict_block(model, block, cfg, rt['device'], rt['names'],
                               bool(cfgvalue(cfg, 'INFER.AMP', True)))
        local = tuple(slice(int(a-b), int(c-b)) for a, c, b in zip(task['start'], task['stop'], low))
        return np.ascontiguousarray(probs[(slice(None), *local)])


def ordered_bounded_map(executors, function, jobs):
    """At most one queued job per worker, plus the result being committed.

    Yield in input order: stitching and contiguous resume counters stay identical
    even when a later GPU finishes first. Workers never mutate output stores.
    """
    iterator = iter(jobs)
    pending = deque()
    for executor in executors:
        job = next(iterator, None)
        if job is None:
            break
        pending.append((executor, job, executor.submit(function, job)))
    try:
        while pending:
            executor, job, future = pending.popleft()
            result = future.result()
            following = next(iterator, None)
            if following is not None:
                pending.append((executor, following, executor.submit(function, following)))
            yield job, result
    finally:
        for _, _, future in pending:
            future.cancel()


class BlockPool:
    def __init__(self, cfg, run_id, workflow):
        import torch
        size = torch.cuda.device_count()
        configure_threads(cfg, size)
        self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                          initializer=_initialize, initargs=(gpu, size, cfg, run_id, workflow))
                          for gpu in range(size)]

    def map(self, jobs):
        return ordered_bounded_map(self.executors, _predict, jobs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        for executor in self.executors:
            if exc_type is not None:
                executor.terminate_workers()
            else:
                executor.shutdown(wait=True, cancel_futures=True)
