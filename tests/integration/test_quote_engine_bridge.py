import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_wrapper_sizes_at_quote_and_reconciles_both_sides():
    engine=os.environ.get('QUOTE_ENGINE_ROOT')
    recipe=os.environ.get('PINNED_RECIPE_PATH')
    if not engine or not recipe:pytest.skip('explicit corrected quote engine and recipe required')
    script='''
from pathlib import Path
from dataclasses import replace
import math,sys
import numpy as np
from tools.systematic_strategy_matrix import setup,environment,check_costs
from app.multiscale_strategy_search import panel
from app.strategy_scenario_matrix import check_trade
from app.directional_risk_calibration import DirectionalRisk
rec,cfg=setup(Path(sys.argv[1]),Path(sys.argv[2]))
cfg=replace(cfg,execution_price_mode='bid_ask_v3')
for cmd,d in ((1,1),(2,-1)):
    cell=panel('calibration')[0]
    cell.update(drift_before=0.,drift_after=0.,vol_before=0.,vol_after=0.)
    env=environment(cell,cfg,(13,rec['list']['n']),constant_sigma=True,
                    parameters=DirectionalRisk((.5,.5,.2)*2))
    while not env.finished and not env.mask()[cmd]:env.step(np.array([0,.5,.5,.5]))
    e=env.engine;k=env.bar+1
    price=e.entry_fill_price(k,d)
    expected=math.floor(.2*math.floor(e.balance*.1*50/price))
    env.step(np.array([cmd,.5,.5,.5]))
    assert e.pos and e.pos.units==expected
    assert e.pos.entry_price==price
    env.step(np.array([3 if d==1 else 4,.5,.5,.5]))
    t=e.trades[0]
    check_trade(t);check_costs(t,e)
    assert abs(e.balance-e.cfg.capital-t['net_usd'])<1e-7
    assert e.result()['accounting_version']=='quote_side_v3'
print('NATIVE_QUOTE_BRIDGE_PASS')
from app.strategy_repair_episode import run_episode
spec=dict(seed=7,drift_before=.0001,drift_after=-.0001,vol_before=.0001,
    vol_after=.0004,switch=60,transition=8,overlay=.00005,noise='clustered',shock='gap',mirror=False)
for mode in ('ideal','nop','inverted'):
    row=run_episode(spec,cfg,(13,rec['list']['n']),mode=mode)
    assert len(row['decisions'])==239
    if mode=='nop':assert not row['trades'] and row['metrics']['net_profit_usd']==0
    assert row['financial_admission'] is False and row['learning_qualified'] is False
    if mode=='ideal':
        for decision in row['decisions']:
            if decision['command'] in (3,4):
                assert decision['expected']['close_net']>decision['expected']['hold_net']
            if decision['command'] in (1,2):
                i=decision['command']-1
                assert decision['expected']['gains'][i]>=decision['expected']['losses_at_stop'][i]
print('FULL_SYNTHETIC_EPISODES_PASS')
'''
    p=subprocess.run([sys.executable,'-c',script,recipe,engine],
        cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True,timeout=60,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    assert p.returncode==0,p.stdout+p.stderr


def test_grid_resume_preserves_closed_evidence_and_rejects_corruption(tmp_path):
    engine=os.environ.get('QUOTE_ENGINE_ROOT');recipe=os.environ.get('PINNED_RECIPE_PATH')
    if not engine or not recipe:pytest.skip('explicit corrected quote engine and recipe required')
    root=Path(__file__).resolve().parents[2];output=tmp_path/'grid'
    cmd=[sys.executable,'-m','tools.quote_repair_grid','--recipe',recipe,'--engine-root',engine,
         '--design',str(root/'configs/validation/quote_repair_grid.json'),'--output',str(output),
         '--workers','1','--batch-size','2','--limit','2','--storage-reserve-gib','.05']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    a=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    assert a.returncode==0,a.stdout+a.stderr
    closed=(output/'CLOSE.json').read_bytes()
    b=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    assert b.returncode==0,b.stdout+b.stderr
    assert (output/'CLOSE.json').read_bytes()==closed
    artifact=next((output/'batches').glob('*.gz'));artifact.write_bytes(b'corrupt')
    c=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    assert c.returncode!=0 and 'QUOTE_GRID_CLOSED_DATA' in c.stderr
    assert (output/'CLOSE.json').read_bytes()==closed
