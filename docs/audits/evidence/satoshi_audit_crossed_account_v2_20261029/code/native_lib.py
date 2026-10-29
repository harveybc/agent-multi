"""Native world construction + env capture (auditor). Uses the producer's World only to run the
pinned engine; all accounting checks in D_replay.py are independent re-derivations."""
import json, os, sys
from pathlib import Path
AM = Path(os.environ['AM']); NATIVE = Path(os.environ['STRATEGY_ENGINE_ROOT'])
sys.path.insert(0, str(AM))
from app.zigzag_opportunities import engine
engine(NATIVE)
from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv
import app.short_horizon.expert_guided.absolute as _abs
assert Path(_abs.__file__).resolve() == NATIVE/'app/short_horizon/expert_guided/absolute.py'
from app.crossed_account_panel import CrossedWorld, execution_specs, market
from app.dynamic_account_arbiter import decode
from tools.optimize_dynamic_account import controls
CONFIG = json.loads((AM/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
CAPTURED = []
_orig_reset = AbsoluteBarEnv.reset
def _reset(self):
    CAPTURED.append(self); return _orig_reset(self)
AbsoluteBarEnv.reset = _reset

def policies(front_rows, reps):
    out = {n: (list(g), e) for n, g, e in controls()}
    genes = {r['key']: r['genes'] for r in front_rows}
    for k in reps:
        out[k] = (genes[k], None)
        if genes[k][8] != 1.:
            f = list(genes[k]); f[8] = 1.; out[k+'_size1'] = (f, None)
    return out

def run_capture(world, genes, enabled, verify=True, details=True):
    CAPTURED.clear()
    res = world.run(genes, enabled, verify, details)
    envs = [e for e in CAPTURED if e is not world.env]
    CAPTURED.clear()
    return res, envs[-1]
