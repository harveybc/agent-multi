import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('side',[1,-1])
def test_native_exit_menu_has_interior_optimum_and_no_post_stop_choices(side):
    engine=os.environ.get('QUOTE_ENGINE_ROOT');recipe=os.environ.get('PINNED_RECIPE_PATH')
    if not engine or not recipe:pytest.skip('explicit engine and recipe required')
    script='''
from pathlib import Path
import sys,numpy as np
from tools.systematic_strategy_matrix import setup
from app.optimal_exit_corpus import build_case
from app.optimal_exit import targets,outcome
rec,cfg=setup(Path(sys.argv[1]),Path(sys.argv[2]))
spec=dict(id='literal',split='train',direction=int(sys.argv[3]),peak=8,gain=.5,offset=0.,spread=1.)
case,x=build_case(spec,cfg,(13,rec['list']['n']))
t=targets(case['close_net'],case['terminal_net'],case['risk_usd'])
i=int(np.argmax(case['close_net']))
assert 0<i<len(x)-1
assert case['terminal_reason']=='SL'
assert np.all(np.diff(case['bars'])>0)
assert x.shape==(len(case['close_net']),13,rec['list']['n'])
assert not np.any(x[:,:,3:])
assert outcome(case,t['oracle_close'])['regret_risk']==0.
assert outcome(case,np.ones(len(x)))['regret_risk']>0.
assert outcome(case,np.zeros(len(x)))['regret_risk']>0.
print('NATIVE_OPTIMAL_EXIT_PASS',spec['direction'],i,len(x))
'''
    r=subprocess.run([sys.executable,'-c',script,recipe,engine,str(side)],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=60,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert r.returncode==0,r.stdout+r.stderr
