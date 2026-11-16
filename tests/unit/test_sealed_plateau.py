import shutil

import numpy as np
import pytest
import torch

from tools.seal_sac_recovery import seal, sha
from tools.run_discrete_curve_probe import write, read
from tools.run_sealed_plateau_continuation import prepare_branch, PLATEAU
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def test_seal_survives_source_pruning_and_identity_fork_preserves_bytes(tmp_path):
    source=tmp_path/'source';gen=source/'recovery'/'epoch_0040';gen.mkdir(parents=True)
    state=dict(identity='old',epoch=40,steps=100,updates=24,history=[],best_bundle=None)
    write(gen/'STATE.json',state)
    (gen/'model.pt').write_bytes(b'model');(gen/'learning.pt').write_bytes(b'replay-rng')
    write(gen/'MANIFEST.json',dict(identity='old',files={p.name:sha(p) for p in gen.iterdir()}))
    write(source/'RESUME.json',dict(directory=gen.name,manifest_sha256=sha(gen/'MANIFEST.json')))
    for name in ('initial_bundle','bundles'):(source/name).mkdir()
    for name in ('DESIGN.json','EFFECTIVE.json','EPOCH_ZERO.json'):write(source/name,{})
    dest=tmp_path/'seal';r=seal(source,dest);shutil.rmtree(source)
    assert seal(source,dest)==r
    prepare_branch(dest,tmp_path/'fork','new',{'budget':10})
    fork=tmp_path/'fork'/'recovery'/gen.name
    assert (fork/'model.pt').read_bytes()==b'model'
    assert (fork/'learning.pt').read_bytes()==b'replay-rng'
    s=read(fork/'STATE.json');assert s.pop('identity')=='new'
    assert s=={k:v for k,v in state.items() if k!='identity'}
    assert read(dest/'recovery'/gen.name/'STATE.json')['identity']=='old'
    (dest/'recovery'/gen.name/'model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError,match='SEALED_FILE_CHANGED'):seal(source,dest)


def test_scheduler_integrated_memmap_restart_after_reduction_exact(tmp_path,request):
    from tests.unit.test_double_tcn_probe import probe_recipe
    from app.three_signal_learning import synthetic_env
    from app.sac_training_reliability import split_optimizer
    from app.g3_replay import JointReplay
    from app.training_forensics import tree_digest
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)
    recipe=probe_recipe();base=request.getfixturevalue('curve_env')
    factory=lambda:synthetic_env(recipe,base.risk_config,101,1,.99)[0]
    def assess(m):
        return {p:dict(return_per_week=1.,orders_per_week=3.) for p in ('TRAIN','VALIDATION')}
    def make():
        m=ThreeSignalSAC(recipe,seed=7)
        split_optimizer(m,dict(actor=1.,critic=1.,temperature=1.))
        return m
    design=dict(replay_size=64,batch_size=16,warmup=16,update_every=4,max_epochs=10,
        minimum_epochs=0,patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=1,
        initialize_epoch_zero=True,fork_epoch=0,actor_plateau=PLATEAU,replay_backend='memmap')
    probe=factory().inputs.windows[30:34].copy()
    kwargs=dict(replay_factory=lambda real:JointReplay(real,None,16,207))
    a=fit_epochs(make(),factory,assess,design,7,tmp_path/'a','same',probe,**kwargs)
    fit_epochs(make(),factory,assess,dict(design,max_epochs=8),7,tmp_path/'b','same',probe,**kwargs)
    b=fit_epochs(make(),factory,assess,design,7,tmp_path/'b','same',probe,**kwargs)
    assert a['actor_plateau']==b['actor_plateau']
    assert a['stopping']==b['stopping'] and a['best_bundle']['path']=='epoch_0000'
    assert a['history'][7]['actor_plateau']['reduced']
    assert a['history'][8]['learning_rates_used']==[.00015,.0003,.0003]
    assert [h['mean_update'] for h in a['history']]==[h['mean_update'] for h in b['history']]
    for file in ('model.pt','learning.pt'):
        left=torch.load(tmp_path/'a'/'recovery'/'epoch_0010'/file,weights_only=False)
        right=torch.load(tmp_path/'b'/'recovery'/'epoch_0010'/file,weights_only=False)
        assert tree_digest(left)==tree_digest(right)
