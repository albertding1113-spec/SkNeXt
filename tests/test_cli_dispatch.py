"""Exercise the real CLI/config dispatcher without loading GPU dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import ast, json, os, sys, types
from pathlib import Path
from sknext.utils.progress import log_message
torch = types.ModuleType('torch')
torch.device = lambda name: name
torch.cuda = types.SimpleNamespace(is_available=lambda: True, set_device=lambda device: None,
                                   get_device_name=lambda device: 'test GPU')
sys.modules['torch'] = torch
utils = types.ModuleType('sknext.utils.utils')
utils.set_seed = lambda: None
sys.modules[utils.__name__] = utils
events = []
def record(kind, cfg, run_id, device, **kwargs):
    events.append({'kind': kind, 'channels': list(cfg['TASK']['CHANNELS']),
                   'weights': list(cfg['TASK']['CHANNEL_WEIGHTS']),
                   'threshold': cfg['INFER']['THRESHOLD'],
                   'run_id': run_id, 'device': device,
                   'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'),
                   'result': cfg['PATHS']['RESULT_DIR']['PATH_']})
semantic = types.ModuleType('sknext.run.semantic_workflow')
semantic.__dict__.update(torch=torch, log_message=log_message)
tree = ast.parse(Path('sknext/run/semantic_workflow.py').read_text(encoding='utf-8'))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in {'cfgget', 'checked_semantic_names', 'run'}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'semantic_workflow.py', 'exec'), semantic.__dict__)
semantic.train = lambda *args, **kwargs: record('semantic_train', *args, **kwargs)
semantic.infer = lambda *args, **kwargs: record('semantic_infer', *args, **kwargs)
sys.modules[semantic.__name__] = semantic
def runner(kind):
    class Runner:
        def __init__(self, cfg, device, run_id):
            self.cfg, self.device, self.run_id = cfg, device, run_id
        def run(self):
            record(kind, self.cfg, self.run_id, self.device)
    return Runner
for module, attribute, kind in (
    ('sknext.run.restoration_workflow', 'RestorationWorkflow', 'restoration'),
    ('sknext.run.networkflow', 'Segmentation_Workflow', 'segmentation'),
):
    stub = types.ModuleType(module)
    setattr(stub, attribute, runner(kind))
    sys.modules[module] = stub
selector = sys.argv[2]
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1'
sys.argv = ['sknext', '--config', sys.argv[1], '--run_id', '9', '--gpu', selector]
from sknext import main
try:
    main()
except SystemExit as exc:
    if exc.code not in (None, 0): raise
print('DISPATCH_RESULT=' + json.dumps(events))
'''


class CliDispatchTests(unittest.TestCase):
    def launch(self, cfg, expected=None, error=None, gpu='GPU-selected'):
        with tempfile.TemporaryDirectory() as folder:
            cfg.setdefault('PATHS', {}).setdefault('RESULT_DIR', {})['PATH'] = folder
            path = Path(folder) / 'config.yaml'
            path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
            proc = subprocess.run([sys.executable, '-c', PROBE, str(path), gpu], cwd=ROOT,
                                  capture_output=True, text=True, timeout=30)
            if error:
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn(error, proc.stderr)
                return
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            line = next(line for line in proc.stdout.splitlines() if line.startswith('DISPATCH_RESULT='))
            events = json.loads(line.split('=', 1)[1])
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event['kind'], expected)
            self.assertEqual(event['run_id'], 9)
            self.assertEqual(event['device'], 'cuda:0')
            self.assertEqual(event['visible_gpu'], '3,2,1' if gpu == 'all' else gpu)
            self.assertEqual(Path(event['result']), Path(folder) / 'results' / '9')
            return event

    def test_multi_gpu_selectors_dispatch_every_workflow(self):
        for gpu in ('0,1', '0,1,2', 'all'):
            for channels, kind in ((['F', 'P'], 'segmentation'),
                                   (['S.a'], 'semantic'), (['SR'], 'restoration')):
                for training in (True, False):
                    with self.subTest(gpu=gpu, channels=channels, training=training):
                        expected = kind + ('_train' if training else '_infer') if kind == 'semantic' else kind
                        self.launch({'TASK': {'CHANNELS': channels},
                                     'TRAIN': {'ENABLE': training}, 'INFER': {'ENABLE': not training}},
                                    expected, gpu=gpu)

    def test_all_bundled_examples_route_through_cli(self):
        for path in sorted((ROOT / 'sknext/example').glob('*.yaml')):
            with self.subTest(config=path.name):
                cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                channels = cfg['TASK']['CHANNELS']
                if channels == ['SR']:
                    expected = 'restoration'
                elif all(ch.startswith('S.') for ch in channels):
                    expected = 'semantic_train' if cfg['TRAIN']['ENABLE'] else 'semantic_infer'
                else:
                    expected = 'segmentation'
                self.launch(cfg, expected)

    def test_semantic_defaults_and_per_channel_thresholds(self):
        event = self.launch({'TASK': {'CHANNELS': ['S.a', 'S.b', 'S.c']},
                             'TRAIN': {'ENABLE': True}}, 'semantic_train')
        self.assertEqual(event['weights'], [1., 1., 1.])
        self.assertEqual(event['threshold'], .5)
        event = self.launch({'TASK': {'CHANNELS': ['S.a', 'S.b'], 'CHANNEL_WEIGHTS': [2, 1]},
                             'INFER': {'ENABLE': True, 'THRESHOLD': [.3, .7],
                                       'SAVE_PROBABILITIES': True}}, 'semantic_infer')
        self.assertEqual(event['threshold'], [.3, .7])
        self.assertEqual(event['weights'], [2, 1])

    def test_invalid_modes_and_mixed_sr_are_rejected(self):
        for channels in (['S.a'], ['SR'], ['F', 'P']):
            for enabled in (False, True):
                with self.subTest(channels=channels, enabled=enabled):
                    self.launch({'TASK': {'CHANNELS': channels},
                                 'TRAIN': {'ENABLE': enabled}, 'INFER': {'ENABLE': enabled}},
                                error='Choose exactly one')
        self.launch({'TASK': {'CHANNELS': ['SR', 'S.a']}, 'TRAIN': {'ENABLE': True}},
                    error='cannot be combined')
