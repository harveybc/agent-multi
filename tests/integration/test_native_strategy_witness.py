import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_exact_frozen_business_scenarios_and_oracle():
    root=os.environ.get('CORRECTED_ENGINE_ROOT')
    campaign=os.environ.get('NATIVE_CAMPAIGN_ROOT')
    if not root or not campaign:pytest.skip('Explicit frozen engine and campaign required')
    script=r'''
import json,sys
from pathlib import Path
from app.recipe_runtime import attach_engine
attach_engine(sys.argv[1])
from app.short_horizon.expert_guided.env import EnvConfig
from app.short_horizon.execution import Sizing,RealisticProfile,OrderRules
from tools.native_strategy_contract_study import witnesses,oracle
from app.native_strategy_contract import RiskParameters
r=json.loads((Path(sys.argv[2])/'RECIPE.json').read_text())
c=dict(r['business']['evaluation_and_policy'])
c.update(sizing=Sizing(**c['sizing']),profile=RealisticProfile(**c['profile']),rules=OrderRules(**c['rules']))
cfg=EnvConfig(**c)
rows=witnesses(cfg,(13,r['list']['n']))
assert len(rows)==22 and all(len(x['trades'])==1 for x in rows)
for amplitude,weeks,phase in ((.002,8,0),(.0015,6,3)):
 waves=oracle(cfg,(13,r['list']['n']),amplitude,weeks,phase)
 assert len(waves)==3
 assert waves[0]['metrics']['net_profit_usd']>0
 assert waves[1]['metrics']['net_profit_usd']<0
 assert waves[2]['metrics']['trades']==0
 assert waves[0]['metrics']['n_long']>0 and waves[0]['metrics']['n_short']>0
baseline=oracle(cfg,(13,r['list']['n']),.002,8,0)[0]['trades'][0]
changed=oracle(cfg,(13,r['list']['n']),.002,8,0,RiskParameters(.75,.25,.2))[0]['trades'][0]
assert changed['entry_time']==baseline['entry_time']
assert changed['units']<baseline['units']
assert changed['percentual_order_size']==.2
assert changed['stop_loss']!=baseline['stop_loss']
assert changed['take_profit']!=baseline['take_profit']
print('EXACT_CONFIG_22_SCENARIOS_AND_TWO_ORACLE_CONTROLS_PASS')
'''
    p=subprocess.run([sys.executable,'-c',script,root,campaign],cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        text=True,capture_output=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
