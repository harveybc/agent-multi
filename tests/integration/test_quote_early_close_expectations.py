"""Non-vacuous, literal two-sided CLOSE/HOLD witnesses on synthetic paths."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('direction', [1, -1])
@pytest.mark.parametrize('continuation', ['stop', 'target'])
def test_ideal_head_closes_before_stop_and_holds_before_target(direction, continuation):
    engine = os.environ.get('QUOTE_ENGINE_ROOT')
    recipe = os.environ.get('PINNED_RECIPE_PATH')
    if not engine or not recipe:
        pytest.skip('explicit corrected quote engine and recipe required')
    script = '''
from pathlib import Path
from dataclasses import replace
import sys
import numpy as np
from tools.systematic_strategy_matrix import setup, environment, clone, action_from_head, check_costs
from app.multiscale_strategy_search import panel
from app.directional_risk_calibration import DirectionalRisk
from app.strategy_repair_episode import ideal_command
from app.strategy_scenario_matrix import check_trade

rec,cfg=setup(Path(sys.argv[1]),Path(sys.argv[2]))
d=int(sys.argv[3]); continuation=sys.argv[4]
cfg=replace(cfg,execution_price_mode='bid_ask_v3')
cell=panel('calibration')[0]
cell.update(drift_before=0.,drift_after=0.,vol_before=0.,vol_after=0.)
env=environment(cell,cfg,(13,rec['list']['n']),constant_sigma=True,
                parameters=DirectionalRisk((.5,.5,.2)*2))
opening=1 if d==1 else 2
while not env.finished and not env.mask()[opening]:
    env.step(np.array([0,.5,.5,.5]))
env.step(np.array([opening,.5,.5,.5]))
e=env.engine
assert e.pos is not None and e.pos.direction==d and not e.trades
p=e.pos;k=env.bar+1
assert e.v[k+1]+3600*10**9<e.ff[e.seg_of[k]]

# Literal synthetic future: safe first bar, then only the specified barrier.
# Construct executable liquidation OHLC and translate to midpoint inputs.
for name in ('o','h','l','c'):
    setattr(e,name,getattr(e,name).copy())
centre=p.entry_price
span=min(abs(centre-p.tp),abs(centre-p.sl))*.1
for j in (k,k+1):
    spread=e.cfg.profile.spread_at(int(e.v[j]),int(e.mo[e.seg_of[j]]))
    end=centre if j==k else (p.sl-d*.0002 if continuation=='stop' else p.tp+d*.0002)
    e.o[j]=centre+d*spread/2
    e.h[j]=max(centre+span,end)+d*spread/2
    e.l[j]=min(centre-span,end)+d*spread/2
    e.c[j]=end+d*spread/2

chosen,reference=ideal_command(env,2)
expected=(3 if d==1 else 4) if continuation=='stop' else 0
assert chosen==expected, (chosen,expected,reference)
assert (reference['close_net']>reference['hold_net'])==(continuation=='stop')
assert reference['reference']['reason']==('SL' if continuation=='stop' else 'TP')
assert reference['reference']['exit_bar']==k+1

# Execute BOTH alternatives on the identical path, not just reference arithmetic.
hold=clone(env)
hold.step(np.array([0,.5,.5,.5]))
assert hold.engine.pos is not None and not hold.engine.trades
hold.step(np.array([0,.5,.5,.5]))
assert hold.engine.pos is None and len(hold.engine.trades)==1
held=hold.engine.trades[0]
assert held['close_reason']==('SL' if continuation=='stop' else 'TP')
assert abs(held['net_usd']-reference['hold_net'])<1e-7

close=clone(env)
close.step(np.array([3 if d==1 else 4,.5,.5,.5]))
assert close.engine.pos is None and len(close.engine.trades)==1
closed=close.engine.trades[0]
assert closed['close_reason']=='AGENT_CLOSE' and closed['direction']==d
assert closed['exit_bar_abs']==k
assert abs(closed['net_usd']-reference['close_net'])<1e-7
assert (closed['net_usd']>held['net_usd'])==(continuation=='stop')

# Inject the known desired categorical scores through the actual output codec.
scores=np.full(5,-1.);scores[chosen]=1.
action=action_from_head(None,env._observe(),scores)
assert int(action[0])==expected
env.step(action)
if continuation=='stop':
    assert env.engine.pos is None and len(env.engine.trades)==1
    assert env.engine.trades[0]['close_reason']=='AGENT_CLOSE'
else:
    assert env.engine.pos is not None and not env.engine.trades
for t,eng in ((closed,close.engine),(held,hold.engine)):
    check_trade(t);check_costs(t,eng)
print('LITERAL_EARLY_CLOSE_PASS', d, continuation)
'''
    result = subprocess.run(
        [sys.executable, '-c', script, recipe, engine, str(direction), continuation],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
             'OPENBLAS_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout + result.stderr
