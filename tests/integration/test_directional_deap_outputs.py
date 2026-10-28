import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_real_engine_directional_heads_and_zero_volume():
    root=os.environ.get('CORRECTED_ENGINE_ROOT');campaign=os.environ.get('NATIVE_CAMPAIGN_ROOT')
    if not root or not campaign:pytest.skip('Pinned engine and campaign required')
    script=r'''
import json,sys
from pathlib import Path
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import EnvConfig
from app.short_horizon.execution import Sizing,RealisticProfile,OrderRules
from app.directional_risk_calibration import DirectionalOrders,DirectionalRisk
from tools.native_strategy_contract_study import oracle
r=json.loads((Path(sys.argv[2])/'RECIPE.json').read_text())
c=dict(r['business']['evaluation_and_policy'])
c.update(sizing=Sizing(**c['sizing']),profile=RealisticProfile(**c['profile']),rules=OrderRules(**c['rules']))
cfg=EnvConfig(**c);shape=(13,r['list']['n'])
original=oracle(cfg,shape,.002,8,0)
def run(genes):
 return oracle(cfg,shape,.002,8,0,DirectionalRisk(tuple(genes)),DirectionalOrders,False)
assert run([.5,.5,1]*2)==original
zero=run([0,1,0,1,0,0])
assert all(x['metrics']['trades']==0 and x['metrics']['net_profit_usd']==0 for x in zero)
assert all(x['metrics'][k]==0 for x in zero for k in ('cost_spread_usd','cost_commission_usd','cost_financing_usd'))
asym=run([0,1,.2,1,0,1])
trades=asym[0]['trades']
assert {t['direction'] for t in trades}=={1,-1}
balance=cfg.capital
for t in trades:
 q=.2 if t['direction']==1 else 1
 assert t['percentual_order_size']==q
 assert t['margin_usd']<=q*cfg.sizing.rel_volume*balance+1e-5
 if t['direction']==1: assert t['stop_loss']<t['entry_price']<t['take_profit']
 else: assert t['take_profit']<t['entry_price']<t['stop_loss']
 balance+=t['net_usd']
print('DIRECTIONAL_ENDPOINTS_ZERO_VOLUME_AND_MIDPOINT_PARITY_PASS')
'''
    p=subprocess.run([sys.executable,'-c',script,root,campaign],cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
