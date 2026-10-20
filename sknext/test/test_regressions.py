"""Dependency-light regression tests for corrected SkNeXt project components.

Run: python tests/test_regressions.py
Requires: numpy, scipy, scikit-image, PyTorch (no Zarr/Navis/YACS required).
"""
import ast
import json
import os
import sys
import math
import random
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from scipy.ndimage import generate_binary_structure

ROOT = Path(__file__).resolve().parents[1] / 'sknext'

def get_function(relpath, name, globals_dict):
    tree = ast.parse((ROOT / relpath).read_text())
    nodes = [x for x in tree.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(relpath), 'exec'), globals_dict)
    return globals_dict[name]

def get_method(relpath, klass, name, globals_dict):
    tree = ast.parse((ROOT / relpath).read_text())
    c = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == klass)
    node = next(x for x in c.body if isinstance(x, ast.FunctionDef) and x.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(relpath), 'exec'), globals_dict)
    return globals_dict[name]

def test_pairing():
    g={'Path':Path, 're':re}
    pair = get_function('data/imageIO.py', 'pair_raw_and_label_tiffs',g)
    raw = [Path('sample_0002.tif'),Path('sample_0001.tif')]
    gt={'instance':[Path('sample_gt_0001.tif'),Path('sample_gt_0002.tif')],
        'S.soma':[Path('sample_gt_0002.soma.tif'),Path('sample_gt_0001.soma.tif')]}
    rr,gg = pair(raw,gt)
    assert [x.stem for x in rr] == ['sample_0001','sample_0002']
    assert gg['instance'][0].stem=='sample_gt_0001'
    assert gg['S.soma'][0].stem=='sample_gt_0001.soma'
    for bad in [
        {'instance':[Path('sample_gt_0001.tif'),Path('missing_gt_0002.tif')]},
        {'instance':[Path('sample_gt_0001.tif'),Path('sample_0001.tif')]},
    ]:
        try: pair(raw,bad)
        except ValueError: pass
        else: raise AssertionError('Invalid label pair did not raise')
    print('PASS strict raw-label pairing')

def test_skeleton_targets():
    from scipy.ndimage import distance_transform_edt
    g = {'np': np, 'regionprops':regionprops,'ndi':ndi,'generate_binary_structure':generate_binary_structure,
         'skeletonize':skeletonize, 'Literal':__import__('typing').Literal,
         'skeleton_main_branch':lambda x:x}
    method = get_function('data/label.py', 'reconstruct_skeleton_from_mask',g)
    label=np.zeros((15,15,15),dtype=np.uint16)
    label[1:5,1:5,1:5]=1  # even-width failure case
    label[9:13,9:13,9:13]=1 # disconnected same-ID component
    result=method(label,mode='full')
    assert np.count_nonzero(result[1:5,1:5,1:5])>=1
    assert np.count_nonzero(result[9:13,9:13,9:13])>=1
    assert not np.any(result[label==0]); print('PASS skeleton fallback and disconnected instance')

def test_channel_and_watershed():
    capture=[]
    def ws(probs,names,*args,**kwargs):
        capture.append((list(names),probs.copy()))
        return np.full(probs.shape[1:],7,dtype=np.uint16)
    g={'np':np,'time_str':lambda:'', 'watershed_with_sk':ws,
       'post_processing_instance':lambda a,b:a, 'post_processing_semantic':lambda a,b:a}
    def define(*args): pass
    define_channels=get_method('run/networkflow.py','Segmentation_Workflow','define_activations_and_channels',g)
    post_channels=get_method('run/networkflow.py','Segmentation_Workflow','define_postprocess_channels',g)
    validate=get_method('run/networkflow.py','Segmentation_Workflow','validate_watershed_config',g)
    post=get_method('run/networkflow.py','Segmentation_Workflow','_postprocess_block_predictions',g)
    a=SimpleNamespace(channels=['F','S.soma','P'], channels_extra_opts={})
    define_channels(a)
    post_channels(a)
    a.save_segmentation=True; a.skeleton_flag=False; a.segmentation_dtype='uint16'
    a.watershed_seed_channels=['P'];a.watershed_seed_channels_thresh=[.5]
    a.watershed_growth_mask_channels=['F'];a.watershed_growth_mask_channels_thresh=[.5]
    a.watershed_topographic_channel='F';a.inferred_block_num=0;a.total_block_num=1
    a.postprocess_dict={}
    validate(a)
    pred=np.zeros((3,3,3,3),dtype=np.float32)
    pred[0]=.9;pred[1]=.7;pred[2]=.1
    res=post(a,pred,np.array([[0,3]]*3),[])
    assert capture[0][0]==['F','P'] and np.all(capture[0][1][1]==.1)
    assert res.shape==(2,3,3,3) and np.all(res[0]==7) and np.all(res[1]==1)
    a.watershed_growth_mask_channels=[]
    try: validate(a)
    except ValueError: pass
    else: raise AssertionError('missing growth mask allowed')
    a.watershed_growth_mask_channels=['F']
    a.watershed_seed_channels=['S.soma']
    try: validate(a)
    except ValueError: pass
    else: raise AssertionError('semantic seed erroneously accepted')
    print('PASS interleaved instance/semantic indexing and watershed preflight')

def test_normalization():
    g={'np':np,'Literal':__import__('typing').Literal}
    norm=get_function('data/preprocessing.py','normalize_img',g)
    g['normalize_img']=norm
    prep=get_function('data/preprocessing.py','preprocess_img',g)
    source=np.random.default_rng(7).normal(size=(1,8,8,8)).astype('float32')
    options={'normalization':True,'norm_type':'zero_mean_unit_variance'}
    a=prep(source.copy(),options)
    b=prep(source.copy(),options)
    assert np.array_equal(a,b) and abs(float(a.mean()))<1e-5
    print('PASS deterministic identical per-patch normalization')

def test_inference_resume():
    import hashlib
    from datetime import datetime
    g={'Path':Path, 'np':np,'os':os, 'json':json,'datetime':datetime,
       'hashlib':hashlib, 'time_str':lambda: ''}
    save=get_method('run/networkflow.py','Segmentation_Workflow','save_infer_log',g)
    load=get_method('run/networkflow.py','Segmentation_Workflow','load_infer_log',g)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        a=SimpleNamespace(job_id=1,infer_log_path=p/'logs',cfg=SimpleNamespace(DATA=SimpleNamespace(INFER=SimpleNamespace(INFER_LOG=str(p/'logs'))), INFER=SimpleNamespace(RAW_ALL_BLOCKS=False)),
                          inferred_block_num=4,total_block_num=9,output_mode='uint16',save_segmentation=True,
                          save_raw_predictions=False,skeleton_flag=True,segmentation_dtype='uint16',
                          next_instance_id=1,instance_num=2,
                          infer_gt_path=p/'output.zarr',infer_raw_path=p/'prob.zarr',
                          infer_manifest={'patch':[8,64,64],'checkpoint':'abc'})
        a.infer_gt_path.mkdir()
        save(a)
        a.inferred_block_num=0
        load(a)
        assert a.inferred_block_num==4
        a.infer_manifest={'checkpoint':'changed'}
        try: load(a)
        except ValueError: pass
        else: raise AssertionError('manifest mismatch allowed')
        (p/'logs'/'infer_log_01.json').unlink()
        try: load(a)
        except FileExistsError: pass
        else: raise AssertionError('output without progress metadata allowed')
    print('PASS crash-safe resume manifest integrity checks')

def test_full_checkpoint():
    import hashlib
    g={'Path':Path,'np':np,'torch':torch,'os':os,'random':random,'time_str':lambda:'',
       'hashlib':hashlib}
    save=get_method('run/networkflow.py','Segmentation_Workflow','save_training_state',g)
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import restore_training_checkpoint
    g['restore_training_checkpoint'] = restore_training_checkpoint
    from sknext.utils.checkpoint import training_checkpoint_paths
    g['training_checkpoint_paths'] = training_checkpoint_paths
    g['get_cfg_value'] = lambda cfg, key, default=False: default
    load=get_method('run/networkflow.py','Segmentation_Workflow','load_training_state',g)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'ckpt_latest.pth'
        original=torch.nn.Linear(2,1)
        opt=torch.optim.AdamW(original.parameters(),lr=.01)
        sch=torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=.9)
        opt.zero_grad(); original(torch.ones(1,2)).sum().backward(); opt.step(); sch.step()
        class Dummy:
            latest_checkpoint_file=p
            checkpoint_file=p.with_name('ckpt.pth')
            checkpoint_input_file=p.with_name('ckpt.pth')
            output_channel_names=[]
            cfg=SimpleNamespace()
            _training_identity=lambda self:'contract'
            model_name='toy'
            epochs=10
            _atomic_torch_save=staticmethod(lambda state, dest: torch.save(state, dest))
        obj=Dummy(); obj.model=original; obj.optimizer=opt;obj.scheduler=sch
        obj.best_val_loss=.23; obj._chart_history={'Loss':{'val':[(1,.23)]}}
        save(obj, 1, {'F':.88}, 0)
        new=Dummy();new.model=torch.nn.Linear(2,1)
        new.optimizer=torch.optim.AdamW(new.model.parameters(),lr=.01)
        new.scheduler=torch.optim.lr_scheduler.StepLR(new.optimizer,1,.9)
        e, metrics, n=load(new)
        assert (e,metrics,n)==(1,{'F':.88},0)
        assert torch.equal(new.model.weight,original.weight)
        assert new.scheduler.state_dict()==sch.state_dict()
        assert len(new.optimizer.state)>0
    print('PASS complete model/optimizer/scheduler/epoch checkpoint roundtrip')

def test_uint16_inference_preserves_float_normalization():
    g={'np':np,'torch':torch}
    basic={'np':np,'Literal':__import__('typing').Literal}
    basic['normalize_img']=get_function('data/preprocessing.py','normalize_img',basic)
    normalize=get_function('data/preprocessing.py','preprocess_img',basic)
    reflect=get_function('data/preprocessing.py','reflect_padding_img',basic)
    g.update({'preprocess_img':normalize, 'reflect_padding_img':reflect})
    batch=get_method('run/networkflow.py','Segmentation_Workflow','_infer_one_batch',g)
    class Capture:
        input=None
        def __call__(self,t):
            self.input=t.detach().clone()
            return torch.zeros((len(t),1,*t.shape[-3:]),dtype=t.dtype)
    model=Capture()
    a=SimpleNamespace(patch_size=(2,2,2,1),device=torch.device('cpu'),batch_size=1,
                      model=model, preprocess_dict={'normalization':True,
                                                     'norm_type':'zero_mean_unit_variance'})
    image=np.arange(8,dtype=np.uint16).reshape((1,2,2,2))*100
    coords=[np.array([[0,2],[0,2],[0,2]])]
    result,weights=batch(a,np.zeros((1,2,2,2),dtype='float32'),
                         np.zeros((2,2,2),dtype='float32'),image,coords,[np.zeros((3,2),dtype=int)])
    assert model.input is not None and model.input.dtype==torch.float32
    assert float(model.input.min())<-.1 and float(model.input.max())>.1
    assert np.all(weights>0)
    print('PASS uint16 microscopy inference retains normalized float32 precision')

def test_swc_mapping():
    g={'np':np,'Path':Path}
    validate=get_method('skeleton/skeleton.py','SkeletonManager','validate_instance_mapping',g)
    class Neu:
        def __init__(self, f,id):self.file=f;self.file_id=id
    a=SimpleNamespace(swc_paths=[Path('0001.swc'),Path('0003.swc')],file_ids=[1,3],
                      skeletons=[Neu(Path('0001.swc'),1),Neu(Path('0003.swc'),3)],
                      id_index_dict={'uuid1':0,'uuid2':1},file_id_index_dict={1:0,3:1})
    rows=validate(a)
    assert [(r['instance_id'],r['swc_file_id']) for r in rows]==[(1,1),(2,3)]
    a.file_ids=[1,1]
    try: validate(a)
    except ValueError: pass
    else:raise AssertionError('broken mapping did not fail')
    print('PASS explicit SWC file ID vs raster label mapping')



def test_legacy_checkpoint_compatibility():
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import (read_checkpoint, load_model_weights,
                                         restore_training_checkpoint)
    import warnings
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original = torch.nn.Linear(3, 2)
        pure = root/'legacy.pth'
        torch.save({'module.'+k:v for k,v in original.state_dict().items()}, pure)
        # Raw DataParallel tensors and absent all optional metadata.
        loaded = torch.nn.Linear(3, 2)
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            report = load_model_weights(loaded, read_checkpoint(pure),
                                        expected_architecture='U_NeXt_V2',
                                        expected_channels=['F','P'])
        assert report['loaded'] == 2 and not report['partial']
        assert torch.equal(loaded.weight, original.weight)
        assert len(log) >= 2  # missing architecture and channel metadata
        # Alternate legacy dictionary wrapper.
        wrapped = root/'state.pth'
        torch.save({'state_dict':original.state_dict()},wrapped)
        loaded2=torch.nn.Linear(3,2)
        assert load_model_weights(loaded2, read_checkpoint(wrapped))['loaded']==2
        # Incomplete model tensor requires explicit opt-in, never silent.
        missing = {'model_state_dict': {'weight': original.weight.detach().clone()}}
        try:
            load_model_weights(torch.nn.Linear(3,2),missing)
        except RuntimeError as exc:
            assert 'CHECKPOINT_ALLOW_PARTIAL' in str(exc)
        else:
            raise AssertionError('Missing tensor must be rejected by default')
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            partial = load_model_weights(torch.nn.Linear(3,2), missing, allow_partial=True)
        assert partial['partial'] and any('PARTIAL CHECKPOINT' in str(w.message) for w in log)
        # Shape mismatch also requires opt-in, even with strict=False implementation.
        wrong = {'model_state_dict':{'weight':torch.randn(6,3),'bias':torch.randn(2)}}
        try: load_model_weights(torch.nn.Linear(3,2), wrong)
        except RuntimeError:pass
        else:raise AssertionError('Wrong tensor shape should fail')
        # Legacy weights-only checkpoint with TRAIN.RESUME=True falls back to epoch 0.
        model = torch.nn.Linear(3,2)
        optim = torch.optim.AdamW(model.parameters(),lr=.01)
        sched = torch.optim.lr_scheduler.StepLR(optim, 1)
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            result = restore_training_checkpoint(
                model, optim, sched, latest_path=root/'absent_latest.pth',
                best_path=pure, identity='current', epochs=20)
        assert result['mode']=='warm_start' and result['epoch']==0
        assert not optim.state
        assert torch.equal(model.weight,original.weight)
        assert any('not an exact resume' in str(w.message) for w in log)
        print('PASS legacy bare/DP/wrapped checkpoints, missing tensor strictness, training fallback')


def test_checkpoint_latest_path_resolution():
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import training_checkpoint_paths
    latest, best=training_checkpoint_paths(Path('checkpoint_01_latest.pth'))
    assert str(latest)=='checkpoint_01_latest.pth'
    assert str(best)=='checkpoint_01.pth'
    latest,best=training_checkpoint_paths(Path('checkpoint_01.pth'))
    assert str(latest)=='checkpoint_01_latest.pth'
    assert str(best)=='checkpoint_01.pth'
    print('PASS explicit latest input checkpoint path resolution')


def test_incomplete_training_state_compatibility():
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import restore_training_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        source=torch.nn.Linear(2,1)
        weights=source.state_dict()
        latest=p/'latest.pth'
        best=p/'best.pth'
        torch.save({'model_state_dict':weights,'identity':'contract','epoch':7,
                    'optimizer_state_dict':{'broken':'state'}},latest)
        dest=torch.nn.Linear(2,1)
        optimizer=torch.optim.AdamW(dest.parameters(),lr=.02)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1)
        import warnings
        with warnings.catch_warnings(record=True) as log:
            warnings.simplefilter('always')
            result=restore_training_checkpoint(dest, optimizer, scheduler,
                latest_path=latest,best_path=best,identity='contract',epochs=10)
        assert result['mode']=='warm_start' and result['epoch']==0
        assert torch.equal(dest.weight,source.weight)
        assert not optimizer.state
        assert any('start fresh' in str(x.message) for x in log)
        torch.save({'model_state_dict':weights,'identity':'DIFFERENT','epoch':7},latest)
        try: restore_training_checkpoint(dest,optimizer,scheduler,latest_path=latest,
                                         best_path=best,identity='contract',epochs=10)
        except ValueError as e: assert 'identity mismatch' in str(e)
        else: raise AssertionError('identity mismatch must be refused')
        print('PASS incomplete latest training-state warm-start and identity protection')



def test_workflow_checkpoint_entry_points():
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import read_checkpoint, load_model_weights
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        src=torch.nn.Linear(3,2)
        old=p/'best_old.pth'
        torch.save({'model_state_dict':src.state_dict()},old)
        # Main workflow uses an explicitly configured input file and loads
        # tensors even without model_name/output_channel_names.
        g={'Path':Path,'read_checkpoint':read_checkpoint,
           'load_model_weights':load_model_weights,
           'get_cfg_value':lambda cfg,key,default=None:default,
           'time_str':lambda:''}
        read_main=get_method('run/networkflow.py','Segmentation_Workflow','load_checkpoint',g)
        target=torch.nn.Linear(3,2)
        runner=SimpleNamespace(model=target,checkpoint_input_file=old,
                               model_name='unext_v2',output_channel_names=['F','P'],
                               device=torch.device('cpu'),cfg={})
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            report=read_main(runner)
        assert report['loaded']==2 and torch.equal(target.weight,src.weight)
        # Semantic runner uses the old path without redirecting its output file.
        gg={'cfgget':lambda cfg, key,default=None: (
                __import__('functools').reduce(lambda obj,k:obj.get(k,{}) if isinstance(obj,dict) else {},
                  key.split('.'),cfg) or default), 'Path':Path}
        output_path=get_function('run/semantic_workflow.py','checkpoint_path',gg)
        gg['checkpoint_path']=output_path
        input_path=get_function('run/semantic_workflow.py','checkpoint_read_path',gg)
        cfg={'MODEL':{'CHECKPOINT_INPUT_PATH':str(old)},
             'TASK':{'CHANNELS':['S.ER']},
             'PATHS':{'RESULT_DIR':{'PATH':str(p)}}}
        assert input_path(cfg,27)==old
        assert output_path(cfg,27)!=old
        print('PASS real main/semantic checkpoint entry points and safe separate input path')



def test_semantic_checkpoint_legacy_loader():
    sys.path.insert(0, str(ROOT.parent))
    from sknext.utils.checkpoint import read_checkpoint, load_model_weights
    with tempfile.TemporaryDirectory() as tmp:
        model1=torch.nn.Linear(3,2)
        old=Path(tmp)/'checkpoint_old.pth'
        torch.save({'model_state_dict':model1.state_dict()},old)
        g={'Path':Path,'torch':torch,'read_checkpoint':read_checkpoint,
           'load_model_weights':load_model_weights}
        cfgget=get_function('run/semantic_workflow.py','cfgget',g)
        g['cfgget']=cfgget
        names=get_function('run/semantic_workflow.py','checked_semantic_names',g)
        g['checked_semantic_names']=names
        outpath=get_function('run/semantic_workflow.py','checkpoint_path',g)
        g['checkpoint_path']=outpath
        readpath=get_function('run/semantic_workflow.py','checkpoint_read_path',g)
        g['checkpoint_read_path']=readpath
        loader=get_function('run/semantic_workflow.py','load_model_checkpoint',g)
        cfg={'MODEL':{'CHECKPOINT_INPUT_PATH':str(old)},
             'TASK':{'CHANNELS':['S.ER']},
             'PATHS':{'RESULT_DIR':{'PATH':str(tmp)}}}
        model2=torch.nn.Linear(3,2)
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            loader(model2,cfg,42,torch.device('cpu'))
        assert torch.equal(model1.weight,model2.weight)
        # Old checkpoints with actual channel metadata must still reject changes.
        torch.save({'model_state_dict':model1.state_dict(),
                    'semantic_channels':['S.spine']},old)
        try:loader(model2,cfg,42,torch.device('cpu'))
        except ValueError as exc:assert 'channel order' in str(exc)
        else:raise AssertionError('semantic channel permutation not blocked')
        print('PASS semantic-only legacy loader and channel order protection')

if __name__=='__main__':
    for name, fn in sorted(globals().copy().items()):
        if name.startswith('test_') and callable(fn):fn()
    print('ALL REGRESSION TESTS PASSED')
