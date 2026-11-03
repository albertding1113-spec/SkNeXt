"""Exercise SR tiling with lightweight models and real tensor inference."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
import unittest
from unittest.mock import patch

import numpy as np
import torch


class SROverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'sknext/run/restoration_workflow.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        names = {'sr_infer_geometry', '_overlap_voxels', 'crop_padded',
                 'normalize_intensity', 'image_intensity_range'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RestorationWorkflow')
        nodes += [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_sr_block']
        cls.ns = dict(np=np, torch=torch, Sequence=Sequence)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), cls.ns)

    def runner(self, size, overlap, padding, scale=1):
        cfg = SimpleNamespace(DATA=SimpleNamespace(PATCH_SIZE=(*size, 1), INFER=SimpleNamespace(
            CHANNEL=[0], OVERLAP=overlap, PADDING=padding,
            BLOCK_FACTOR=(3,3,3), BLOCK_CENTRAL_FACTOR=(1,1,1))),
            SR=SimpleNamespace(OUTPUT_CLIP=False))
        return SimpleNamespace(cfg=cfg, ch=1, scale=scale, patch=size, device=torch.device('cpu'))

    def predict(self, runner, raw, model, start=None, stop=None, batch=3):
        geometry = self.ns['sr_infer_geometry'](runner.cfg)
        with patch('sknext.utils.parallel.resolve_inference_batch', return_value=batch):
            return self.ns['_predict_sr_block'](runner, model, raw,
                start if start is not None else (0,0,0),
                stop if stop is not None else raw.shape[1:], (0.,1.), geometry)

    def test_identity_preserved_with_padding_scales_and_partial_blocks(self):
        raw = np.linspace(0, 1, 5*9*11, dtype=np.float32).reshape(1,5,9,11)
        for scale in (1, 2, 4):
            for padding in ((0,0,0), (1,1,1)):
                for overlap in ((0,0,0), (.25,.25,.25)):
                    runner = self.runner((4,8,8), overlap, padding, scale)
                    model = lambda x: x.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
                    for start, stop in (((0,0,0), (5,9,11)), ((2,3,4), (5,8,10))):
                        result = self.predict(runner, raw, model, start, stop)
                        expected = raw[:, start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
                        expected = expected.repeat(scale, -2).repeat(scale, -1)
                        np.testing.assert_allclose(result, expected, atol=2e-7)

    def test_disagreeing_predictions_are_blended_not_overwritten(self):
        raw = np.zeros((1,1,1,14), dtype=np.float32)
        runner = self.runner((1,1,8), (0,0,.25), (0,0,0))
        model = lambda x: torch.arange(8, dtype=x.dtype).reshape(1,1,1,1,8).expand_as(x)
        for batch in (1, 3):
            result = self.predict(runner, raw, model, batch=batch)
            np.testing.assert_allclose(result[0,0,0,6:8], [4., 3.], atol=1e-6)
            self.assertTrue(np.isfinite(result).all())

    def test_invalid_overlap_rejected(self):
        for overlap in ((-1,0,0), (1,0,0), (float('nan'),0,0), (.5,0,0)):
            runner = self.runner((4,8,8), overlap, (1,1,1))
            with self.assertRaises(ValueError):
                self.ns['sr_infer_geometry'](runner.cfg)


if __name__ == '__main__':
    unittest.main()
