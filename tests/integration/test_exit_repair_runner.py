import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from app.frozen_recipe_inputs import file_sha
from tools.recipe_sac_queue import write, read
from tests.unit.test_frozen_recipe_inputs import recipe


def test_runner_closes_replays_then_rejects_changed_checkpoint(tmp_path):
    source=tmp_path/'source';out=tmp_path/'out'
    source.mkdir()
    write(source/'DIAGNOSTIC_RECIPE.json',recipe())
    for d in (1,-1):
        case=dict(id=f'train_{d}',close_net=[0.,10.,-5.],terminal_net=-20.,risk_usd=10.,direction=d)
        path=source/'corpus'/f'train_{d}.json';write(path,case)
        np.savez_compressed(path.with_suffix('.npz'),observations=np.arange(18,dtype=np.float32).reshape(3,3,2))
    manifest={str(p.relative_to(source)):file_sha(p) for p in source.rglob('*') if p.is_file()}
    write(source/'CLOSE.json',dict(artifacts=manifest))
    design=dict(schema='exit_learning_repair.v1',capacity_epochs=1,seeds=[7],
        learning=dict(n_steps=16,batch_size=8,n_epochs=1),
        arms=[dict(name='tiny',kind='pure_rl',start='uniform',steps=32)])
    cfg=tmp_path/'design.json';write(cfg,design)
    command=[sys.executable,'-m','tools.exit_learning_repair','--source',str(source),
             '--design',str(cfg),'--output',str(out),'--pilot']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    before=(out/'CLOSE.json').read_bytes()
    assert read(out/'STATUS.json')==read(out/'PROGRESS.json')
    assert read(out/'CLOSE.json')['units']==2
    assert not (out/'CONFIRMATION.json').exists()
    assert subprocess.run(command,env=env,capture_output=True,timeout=60).returncode==0
    assert (out/'CLOSE.json').read_bytes()==before
    model=out/'fits'/'tiny_7'/'model.zip'
    model.write_bytes(model.read_bytes()+b'bad')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0 and 'EXIT_REPAIR_RECEIPT' in result.stderr
    assert (out/'CLOSE.json').read_bytes()==before
