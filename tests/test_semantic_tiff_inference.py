"""Exercise TIFF inference with real TIFF/Zarr I/O and a small CPU predictor.

Load the relevant definitions without importing unrelated training dependencies,
following the dependency-light regression tests elsewhere in this repository.
"""
from __future__ import annotations

import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import Mock

import h5py
import numpy as np
import tifffile
import torch
import zarr
from numcodecs import Blosc


ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)


def runtime():
    ns = dict(globals())
    definitions('sknext/data/datasetIO.py',
                {'ZarrIOManager', 'detect_path_type', '_is_zarr_store', '_has_ims_structure'}, ns)
    definitions('sknext/data/imageIO.py',
                {'get_tif_path_in_folder', 'patch_coordinates_iter'}, ns)
    definitions('sknext/data/preprocessing.py',
                {'preprocess_img', 'normalize_img', 'percentile_clip_img', 'reflect_padding_img'}, ns)
    tree = ast.parse((ROOT / 'sknext/run/semantic_workflow.py').read_text(encoding='utf-8'))
    definitions('sknext/run/semantic_workflow.py',
                {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name not in {'train', 'main', 'make_model'}}, ns)

    class Predictor(torch.nn.Module):
        def forward(self, x):
            return torch.cat([torch.full_like(x[:, :1], 2), torch.full_like(x[:, :1], -2)], dim=1)

    ns.update(make_model=Mock(return_value=Predictor()), load_model_checkpoint=Mock(),
              print_model_parameters=Mock(), print_cuda_memory=Mock(), log_message=Mock())
    return ns


class SemanticTiffInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw = self.base / 'raw'
        self.raw.mkdir()
        self.output = self.base / 'output'
        self.checkpoint = self.base / 'weights.pth'
        self.checkpoint.write_bytes(b'test checkpoint identity')
        self.cfg = {
            'TASK': {'CHANNELS': ['S.ER', 'S.mito']},
            'MODEL': {'CHECKPOINT': str(self.checkpoint)},
            'DATA': {'PATCH_SIZE': [2, 4, 4, 1],
                     'NORMALIZATION': {'ENABLE': False},
                     'INFER': {'PATH': str(self.raw), 'GT_PATH': str(self.output),
                               'CHANNEL': [0], 'BLOCK_CENTRAL_FACTOR': [1, 1, 1],
                               'HALO': [1, 1, 1]}},
            'INFER': {'SAVE_PROBABILITIES': True},
        }
        self.ns = runtime()

    def infer(self):
        self.ns['infer'](self.cfg, 7, torch.device('cpu'))

    def test_folder_prediction_and_resume_use_separate_outputs(self):
        shapes = {'a.tif': (3, 5, 6), 'b.TIFF': (2, 3, 5)}
        for name, shape in shapes.items():
            data = np.ones(shape, dtype=np.uint16)
            # Also exercise compressed, multichannel input and imread fallback.
            tifffile.imwrite(self.raw / name, data if name == 'a.tif' else np.stack([data, data * 2]),
                             compression=None if name == 'a.tif' else 'deflate', photometric='minisblack')
        (self.raw / 'ignore.txt').write_text('not an image')
        (self.raw / 'nested').mkdir()
        self.infer()
        self.ns['make_model'].assert_called_once()
        for name, shape in shapes.items():
            output = self.output / f'{Path(name).stem}_semantic_7.ome.zarr'
            with self.ns['ZarrIOManager'](output, mode='r') as reader:
                self.assertEqual(reader.shape, (2, *shape))
                np.testing.assert_array_equal(reader[0], np.ones(shape, dtype='uint8'))
                np.testing.assert_array_equal(reader[1], np.zeros(shape, dtype='uint8'))
            prob = output.with_name(output.name[:-5] + '_prob.zarr')
            with self.ns['ZarrIOManager'](prob, mode='r') as reader:
                np.testing.assert_allclose(reader[0] / 65535., torch.sigmoid(torch.tensor(2.)).item(), atol=2e-5)
            progress = json.loads(output.with_name(output.name + '.progress.json').read_text())
            self.assertTrue(progress['completed'])
        # A completed folder run must not predict any blocks again.
        self.ns['_predict_block'] = Mock(side_effect=AssertionError('Completed block was recomputed'))
        self.infer()
        self.ns['_predict_block'].assert_not_called()
        changed = self.raw / 'a.tif'
        os.utime(changed, ns=(changed.stat().st_atime_ns, changed.stat().st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(RuntimeError, 'Cannot resume'):
            self.infer()

    def test_parallel_block_results_match_serial_at_volume_edges(self):
        from concurrent.futures import ThreadPoolExecutor
        from sknext.utils.inference_workers import ordered_bounded_map
        raw = np.arange(3*5*6, dtype=np.uint16).reshape(3,5,6) + 1
        tifffile.imwrite(self.raw / 'a.tif', raw, photometric='minisblack')
        self.infer()
        source = self.raw / 'a.tif'
        output = self.base / 'parallel.zarr'
        ns, cfg = self.ns, self.cfg
        model = ns['make_model'](cfg, torch.device('cpu'))
        def predict(job):
            reader, channels = ns['_input_reader'](source, [0])
            try:
                low, high = job['low'], job['high']
                key = tuple(slice(int(a), int(b)) for a,b in zip(low, high))
                block = np.stack([reader[(ch,*key)] for ch in channels])
                probs = ns['_predict_block'](model, block, cfg, torch.device('cpu'), ['S.ER','S.mito'], False)
                local = tuple(slice(int(a-b),int(c-b)) for a,c,b in zip(job['start'],job['stop'],low))
                return np.ascontiguousarray(probs[(slice(None),*local)])
            finally:
                reader.close()
        with ThreadPoolExecutor(2) as executor:
            class Pool:
                def map(self, jobs):
                    return ordered_bounded_map([executor, executor], predict, jobs)
            ns['_infer_source'](None, cfg, 7, torch.device('cpu'), ['S.ER','S.mito'], [0],
                                False, source, output, False, Pool())
        serial = self.output / 'a_semantic_7.ome.zarr'
        for a,b in ((serial, output), (serial.with_name(serial.name[:-5]+'_prob.zarr'),
                                      output.with_name('parallel_prob.zarr'))):
            with ns['ZarrIOManager'](a, mode='r') as first, ns['ZarrIOManager'](b, mode='r') as second:
                np.testing.assert_array_equal(first[:], second[:])

    def test_resume_with_changed_batch_size_preserves_completed_blocks(self):
        tifffile.imwrite(self.raw / 'a.tif', np.ones((3, 5, 6), dtype='uint16'),
                         photometric='minisblack')
        self.cfg['TRAIN'] = {'BATCH_SIZE': 1}
        self.infer()
        output = self.output / 'a_semantic_7.ome.zarr'
        progress_path = output.with_name(output.name + '.progress.json')
        progress = json.loads(progress_path.read_text())
        self.assertEqual(progress['meta']['batch_size'], 1)
        # Simulate an interrupted run with one block left to process.
        progress['completed'].pop()
        progress_path.write_text(json.dumps(progress))
        self.cfg['TRAIN']['BATCH_SIZE'] = 4
        self.ns['_predict_block'] = Mock(wraps=self.ns['_predict_block'])
        self.infer()
        self.ns['_predict_block'].assert_called_once()
        resumed = json.loads(progress_path.read_text())
        self.assertEqual(resumed['meta']['batch_size'], 4)
        self.assertEqual(len(resumed['completed']), len(progress['completed']) + 1)
        # Other inference settings must still match.
        self.cfg['DATA']['INFER']['HALO'] = [0, 0, 0]
        with self.assertRaisesRegex(RuntimeError, 'Cannot resume'):
            self.infer()

    def test_tiff_channel_order_and_single_file_input(self):
        data = np.arange(2 * 3 * 4 * 5, dtype='uint16').reshape(2, 3, 4, 5)
        path = self.raw / 'channels.tif'
        tifffile.imwrite(path, data, photometric='minisblack')
        reader, channels = self.ns['_input_reader'](path, [1, 0])
        try:
            np.testing.assert_array_equal(reader[channels[0]], data[1])
            np.testing.assert_array_equal(reader[channels[1]], data[0])
        finally:
            reader.close()
        self.cfg['DATA']['INFER']['PATH'] = str(path)
        self.infer()
        self.assertTrue((self.output / 'semantic_7.ome.zarr').exists())

    def test_reject_empty_folder_and_duplicate_stems(self):
        with self.assertRaisesRegex(ValueError, 'No TIFF'):
            self.ns['_inference_jobs'](self.cfg, 7)
        for name in ['same.tif', 'same.tiff']:
            (self.raw / name).touch()
        with self.assertRaisesRegex(ValueError, 'unique stems'):
            self.ns['_inference_jobs'](self.cfg, 7)

    def test_reject_store_destination_for_folder(self):
        (self.raw / 'a.tif').touch()
        self.cfg['DATA']['INFER']['GT_PATH'] = str(self.output / 'one.zarr')
        with self.assertRaisesRegex(ValueError, 'output directory'):
            self.ns['_inference_jobs'](self.cfg, 7)

    def test_reject_invalid_tiff_dimensions_and_channel(self):
        path = self.raw / 'a.tif'
        tifffile.imwrite(path, np.ones((4, 5), dtype='uint8'))
        with self.assertRaisesRegex(ValueError, 'ZYX or CZYX'):
            self.infer()
        tifffile.imwrite(path, np.ones((2, 4, 5), dtype='uint8'), photometric='minisblack')
        self.cfg['DATA']['INFER']['CHANNEL'] = [1]
        with self.assertRaisesRegex(ValueError, 'index out of range'):
            self.infer()

    def test_zarr_source_keeps_existing_output_path(self):
        source = self.base / 'input.ome.zarr'
        with self.ns['ZarrIOManager'](source, shape=(1, 2, 4, 5), dtype='uint16') as writer:
            writer[:] = np.ones((1, 2, 4, 5), dtype='uint16')
        self.cfg['DATA']['INFER']['PATH'] = str(source)
        self.assertEqual(self.ns['_inference_jobs'](self.cfg, 7),
                         [(source, self.output / 'semantic_7.ome.zarr')])
        self.infer()
        with self.ns['ZarrIOManager'](self.output / 'semantic_7.ome.zarr', mode='r') as reader:
            np.testing.assert_array_equal(reader[0], np.ones((2, 4, 5), dtype='uint8'))


if __name__ == '__main__':
    unittest.main()
