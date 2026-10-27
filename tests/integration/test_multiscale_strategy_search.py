import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_native_tape_is_identical_across_genes_and_ledger_reconciles():
    recipe=os.environ.get('PINNED_RECIPE_PATH');engine=os.environ.get('PINNED_ENGINE_ROOT')
    if not recipe or not engine:pytest.skip('explicit pinned runtime required')
    code='''
import sys
from pathlib import Path
from app.multiscale_strategy_search import initialize,panel,episode,aggregate
initialize(Path(sys.argv[1]),Path(sys.argv[2]))
rows=[]
for c in panel('calibration'):
    a=episode((c,[.5,.5,1]*2,'ideal'))
    b=episode((c,[.75,.25,.2]*2,'ideal'))
    assert a['tape_sha256']==b['tape_sha256']
    assert abs(sum(t['net_usd'] for t in a['trades'])-a['net_profit_usd'])<1e-7
    rows.append(a)
r=aggregate(rows)
assert r['eligible'] and r['trades']>0 and r['close_reasons']['SL']>0
z=[episode((c,[.5,.5,0]*2,'ideal')) for c in panel('calibration')]
assert aggregate(z)['components']['sharpe'] is None
assert all(not r['trades'] and r['net_profit_usd']==0 for r in z)
print('TAPE_AND_ACCOUNTING_PASS')
'''
    p=subprocess.run([sys.executable,'-c',code,recipe,engine],cwd=Path(__file__).resolve().parents[2],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
        text=True,capture_output=True,timeout=60)
    assert p.returncode==0,p.stdout+p.stderr
