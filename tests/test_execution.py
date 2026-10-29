"""CPU-process coverage of the same sharding, IPC and reductions used on GPUs."""
import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sknext.utils.execution import cpu_budget, local_batch_size
from sknext.utils.inference_workers import ordered_bounded_map


def _delayed_square(job):
    time.sleep(.02 if job % 2 == 0 else .001)
    if job == 5:
        raise RuntimeError('worker failed')
    return job * job


class PairDataset:
    def __len__(self):
        return 7

    def __getitem__(self, index):
        import torch
        return torch.tensor([float(index)]), torch.tensor([2. * index])


def _ddp_probe(rank, rendezvous, output):
    import torch
    import torch.distributed as dist
    from sknext.utils.execution import (make_loader, set_epoch, reduce_sums,
        prepare_once, training_rng_states, validation_model)
    os.environ['SKNEXT_WORLD_SIZE'] = '2'
    os.environ['SKNEXT_RANK'] = str(rank)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        cfg = {'TRAIN': {'BATCH_SIZE': 4, 'NUM_WORKERS': 0}}
        loader = make_loader(PairDataset(), cfg, torch.device('cpu'))
        assert len(loader) == 2 and loader.batch_size == 2
        set_epoch(loader, 3)
        first = list(loader.sampler)
        set_epoch(loader, 4)
        assert first != list(loader.sampler)
        model = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(1, 1, bias=False))
        model.module.weight.data.fill_(1.)
        opt = torch.optim.SGD(model.parameters(), lr=.01)
        # Distinct rank inputs: gradients must match a single global mean loss.
        x = torch.tensor([[float(rank + 1)]])
        model(x).square().mean().backward()
        opt.step()
        torch.testing.assert_close(model.module.weight, torch.tensor([[.95]]))
        validation = make_loader(PairDataset(), cfg, torch.device('cpu'), training=False)
        samples = [float(x) for batch, _ in validation for x in batch[:, 0]]
        totals = reduce_sums([len(samples), sum(samples)], torch.device('cpu'))
        assert totals == [7., 21.], totals
        # Rank 1 has no validation batches: direct module evaluation cannot hang.
        small = make_loader(torch.utils.data.Subset(PairDataset(), [0]), cfg,
                            torch.device('cpu'), training=False)
        for inputs, _ in small:
            validation_model(model)(inputs)
        assert reduce_sums([len(small.dataset)], torch.device('cpu')) == [1.]
        def prepare():
            Path(output).write_text('rank zero only')
            return 'prepared'
        assert prepare_once(prepare) == 'prepared'
        assert len(training_rng_states()) == 2
    finally:
        dist.destroy_process_group()


def _workflow_epoch_probe(rank, rendezvous, folder):
    """Run the production epoch loops on two CPU ranks with a tiny model/data.

    AST loading isolates loop correctness from the installed torchvision ABI and
    expensive TIFF preparation. Real loaders, DDP, optimizers and schedulers run.
    """
    import hashlib
    import json
    import math
    import random
    import numpy as np
    import torch
    import torch.distributed as dist
    import yaml
    from sknext.config.config import SkNeXt_Config
    from sknext.utils import execution as ex
    from sknext.utils.parallel import model_state_dict
    from sknext.utils.utils import get_cfg_value
    from sknext.model.scheduler import build_warmup_cosine_scheduler, build_onecycle_scheduler
    os.environ.update(SKNEXT_WORLD_SIZE='2', SKNEXT_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        root = Path(folder)
        cfg = SkNeXt_Config(job_id=1).get_cfg_defaults()
        cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_WORKERS, cfg.TRAIN.EPOCHS = 4, 0, 2
        cfg.TRAIN.ENABLE, cfg.TRAIN.AMP = True, False
        cfg.TRAIN.RESUME, cfg.MODEL.LOAD_CHECKPOINT = False, False
        cfg.TRAIN.LR_SCHEDULER.WARMUP_COSINE_DECAY_EPOCHS = 0
        cfg.DATA.PATCH_SIZE = (2, 2, 2, 1)
        for part in (cfg.DATA.TRAIN, cfg.DATA.VAL):
            part.PATH = part.GT_PATH = str(root)
        cfg.PATHS.CHECKPOINT_DIR = str(root)
        cfg.PATHS.RESULT_DIR.PATH_ = str(root)
        cfg.PATHS.RESULT_DIR.OUTPUT_CHART = str(root)
        class Data:
            def __init__(self, val=False): self.val = val
            def __len__(self): return 1 if self.val else 7
            def __getitem__(self, i):
                return torch.full((1,2,2,2), (i+1)/10.), torch.ones(1,2,2,2)
        def model():
            return torch.nn.parallel.DistributedDataParallel(torch.nn.Conv3d(1,1,1))
        def saved(path, value):
            assert rank == 0, 'Non-primary rank attempted an output write'
            Path(path).write_text(str(value))
        class Chart:
            def __init__(self, *args): assert rank == 0
            def log(self, *args): assert rank == 0
        ns = dict(torch=torch, np=np, Path=Path, os=os, json=json, math=math,
                  random=random, hashlib=hashlib, time_str=lambda: '', log_message=lambda *a, **k: None,
                  print_model_parameters=lambda *a: None, print_cuda_memory=lambda *a, **k: None,
                  model_state_dict=model_state_dict, TrainingLossChart=Chart,
                  build_warmup_cosine_scheduler=build_warmup_cosine_scheduler,
                  build_onecycle_scheduler=build_onecycle_scheduler, get_cfg_value=get_cfg_value,
                  ZarrPatchLoader=lambda cfg, path, if_val: Data(if_val),
                  loss_log_steps=lambda steps: set())
        ns.update({name: getattr(ex, name) for name in ('prepare_once','make_loader','set_epoch',
                  'primary','reduce_sums','reduce_metrics','world_size','validation_model','NullWriter',
                  'training_rng_states')})
        def definitions(path, names, class_name=None):
            tree = ast.parse(Path(path).read_text(encoding='utf-8'))
            body = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name) if class_name else tree.body
            nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)
        # Semantic: real checkpoint serialization and aggregated validation/IoU.
        definitions('sknext/run/semantic_workflow.py',
                    {'cfgget','checked_semantic_names','normalize_options','checkpoint_path','train'})
        semantic_cfg = yaml.safe_load(cfg.dump())
        semantic_cfg['MODEL']['CHECKPOINT'] = str(root / 'semantic.pth')
        semantic_cfg['TASK'].update(CHANNELS=['S.test'], CHANNEL_WEIGHTS=[1.])
        ns.update(prepare_patch_data=lambda cfg, part, names: root / part,
                  make_model=lambda cfg, device: model(),
                  instance_segmentation_loss=lambda **kw: torch.nn.MSELoss())
        ns['train'](semantic_cfg, 1, torch.device('cpu'))
        # Restoration: uneven validation shard and single-writer checkpoints.
        definitions('sknext/run/restoration_workflow.py', {'train'}, 'RestorationWorkflow')
        ns.update(make_sr_patch_store=lambda cfg, part, path: path,
                  SRPatchDataset=lambda path, **kw: Data('val' in str(path)),
                  sr_loss=lambda pred, y, cfg: (pred-y).square().mean(),
                  sr_psnr=lambda pred, y: 1., SummaryWriter=lambda **kw: ex.NullWriter())
        sr = SimpleNamespace(cfg=cfg, device=torch.device('cpu'), workdir=root,
             latest=root/'sr_latest', checkpoint=root/'sr_best', build_model=model,
             _checkpoint_data=lambda epoch, *a, **kw: epoch,
             _save=lambda data, path: saved(path, data))
        ns['train'](sr)
        # Instance workflow: distributed patch-store metadata, metrics and saves.
        definitions('sknext/run/networkflow.py', {'train'}, 'Segmentation_Workflow')
        runner = SimpleNamespace(cfg=cfg, device=torch.device('cpu'), channels=['F'], job_id=1,
            batch_size=4, patch_size=(2,2,2,1), epochs=2, best_val_loss=float('inf'), patience=0,
            load_checkpoint_flag=False, _prepare_training_stores=lambda: dict(train_zarr_path=root/'train', val_zarr_path=root/'val'),
            _log_scalars=lambda *a: None, save_checkpoint=lambda: saved(root/'seg_best', 'best'),
            save_training_state=lambda epoch, *a: saved(root/'seg_latest', epoch))
        runner.set_model = lambda: setattr(runner, 'model', model())
        def optimizer():
            runner.optimizer = torch.optim.Adam(runner.model.parameters(), lr=.01)
            runner.scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, 1)
        runner.set_optimizer_scheduler = optimizer
        def loss():
            runner.loss_func = torch.nn.MSELoss()
            runner.metric = lambda pred, y: {'F': 1.}
        runner.set_loss_metrics = loss
        ns['train'](runner)
    finally:
        dist.destroy_process_group()


class ExecutionTests(unittest.TestCase):
    def test_instance_worker_initializes_paths_without_loading_unused_skeletons(self):
        from sknext.config.config import SkNeXt_Config
        from sknext.utils import inference_workers as workers
        cfg = SkNeXt_Config(job_id=1).get_cfg_defaults()
        cfg.INFER.RAW_ALL_BLOCKS = True
        skeleton_calls = []
        class Runner:
            def __init__(self, cfg, device, run_id):
                self.infer_path = 'input.zarr'
                self.skeleton_flag = True
                self.save_raw_predictions, self.save_segmentation = True, False
                self.model = SimpleNamespace(eval=lambda: None)
            def set_model(self): pass
            def load_checkpoint(self): pass
            def define_postprocess_channels(self): pass
            def _configure_infer_output(self):
                assert isinstance(self.infer_path, Path)
            def validate_watershed_config(self): pass
            def create_reader(self): pass
            def create_skeleton(self): skeleton_calls.append(True)
        fake = SimpleNamespace(Segmentation_Workflow=Runner)
        with patch.dict('sys.modules', {'sknext.run.networkflow': fake}), \
             patch.dict(os.environ), patch.object(workers, 'configure_threads'), \
             patch('torch.cuda.set_device'), patch.object(workers, '_runtime', None):
            workers._initialize(0, 4, cfg, 1, 'segmentation')
            self.assertEqual(cfg.TRAIN.BATCH_SIZE, 1)
            self.assertEqual(skeleton_calls, [])

    def test_cpu_budget_and_global_batches(self):
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '128'}), patch('os.cpu_count', return_value=128):
            if hasattr(os, 'sched_getaffinity'):
                with patch('os.sched_getaffinity', return_value=set(range(128))):
                    self.assertEqual(cpu_budget(4), 32)
            else:
                self.assertEqual(cpu_budget(4), 32)
        self.assertEqual(local_batch_size(16, 4), 4)
        for total in (0, 1, 6):
            with self.assertRaises(ValueError):
                local_batch_size(total, 4)

    def test_ordered_process_results(self):
        with ProcessPoolExecutor(2, mp_context=mp.get_context('spawn')) as executor:
            results = list(ordered_bounded_map([executor, executor], _delayed_square, range(5)))
        self.assertEqual(results, [(i, i*i) for i in range(5)])

    def test_failure_never_commits_later_blocks_and_prefetch_is_bounded(self):
        generated = []
        def jobs():
            for i in range(100):
                generated.append(i)
                yield i
        committed = []
        with ThreadPoolExecutor(2) as executor:
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                for job, result in ordered_bounded_map([executor, executor], _delayed_square, jobs()):
                    committed.append(job)
                    self.assertLessEqual(len(generated) - len(committed), 2)
        self.assertEqual(committed, list(range(5)))

    def test_spawned_loader_reads_all_samples(self):
        import torch
        from sknext.utils.execution import make_loader
        cfg = {'TRAIN': {'BATCH_SIZE': 3, 'NUM_WORKERS': 2, 'PREFETCH_FACTOR': 1}}
        with patch('sknext.utils.execution.cpu_budget', return_value=16):
            loader = make_loader(PairDataset(), cfg, torch.device('cpu'), training=False)
        actual = [float(x) for batch, _ in loader for x in batch[:, 0]]
        self.assertEqual(actual, list(range(7)))
        # Persistent workers can run the next epoch without stale or missing data.
        self.assertEqual(sum(len(batch) for batch, _ in loader), 7)

    def test_two_process_gradient_reduction_and_uneven_validation(self):
        import torch.multiprocessing as tmp
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = (Path(folder) / 'init').resolve().as_uri()
            output = str(Path(folder) / 'prepared.txt')
            tmp.spawn(_ddp_probe, args=(rendezvous, output), nprocs=2, join=True)
            self.assertEqual(Path(output).read_text(), 'rank zero only')

    def test_all_workflow_epoch_loops_on_two_ranks(self):
        import torch.multiprocessing as tmp
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = (Path(folder) / 'init').resolve().as_uri()
            tmp.spawn(_workflow_epoch_probe, args=(rendezvous, folder), nprocs=2, join=True)
            self.assertEqual((Path(folder)/'sr_latest').read_text(), '2')
            self.assertEqual((Path(folder)/'seg_latest').read_text(), '2')
            self.assertTrue(list(Path(folder).glob('*latest.pth')))

    def test_restoration_block_batching_matches_identity_at_edges(self):
        import numpy as np
        import torch
        from typing import Sequence
        tree = ast.parse(Path('sknext/run/restoration_workflow.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RestorationWorkflow')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_sr_block')
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 {'crop_padded', 'normalize_intensity', 'image_intensity_range'}]
        ns = dict(np=np, torch=torch, Sequence=Sequence)
        exec(compile(ast.Module(body=nodes+[method], type_ignores=[]), '<sr-test>', 'exec'), ns)
        cfg = {'TRAIN': {'BATCH_SIZE': 3}, 'INFER': {'BATCH_SIZE': 0}}
        class Config(dict):
            __getattr__ = dict.__getitem__
        cfg = Config(cfg)
        cfg.DATA = SimpleNamespace(INFER=SimpleNamespace(CHANNEL=[0]))
        cfg.SR = SimpleNamespace(OUTPUT_CLIP=False)
        cfg.TRAIN = SimpleNamespace(BATCH_SIZE=3)
        runner = SimpleNamespace(cfg=cfg, ch=1, scale=1, patch=(4,4,4), device=torch.device('cpu'))
        raw = np.linspace(0., 1., 5*7*9, dtype=np.float32).reshape(1,5,7,9)
        geometry = {'padding': np.array([1,1,1]), 'patch_central': np.array([2,2,2]),
                    'block_margin': np.array([1,1,1])}
        batches = []
        class Identity(torch.nn.Module):
            def forward(self, x):
                batches.append(len(x))
                return x
        for start, stop in (([0,0,0],[3,5,5]), ([3,5,5],[5,7,9])):
            result = ns['_predict_sr_block'](runner, Identity(), raw, start, stop, (0.,1.), geometry)
            expected = raw[(slice(None), *[slice(a,b) for a,b in zip(start,stop)])]
            np.testing.assert_allclose(result, expected)
        self.assertIn(3, batches)
        self.assertTrue(any(n < 3 for n in batches))
