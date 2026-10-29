import json, sys, gzip
from pathlib import Path
import os; AM=Path(os.environ.get('AM_WORKTREE', '<am-worktree>')); HS=Path(os.environ.get('HS_ENGINE_ROOT', '<hs-engine-pin>'))
sys.path.insert(0,str(AM))
from app.zigzag_opportunities import engine
engine(HS)
CFG=json.loads((AM/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
def replays(run):
    return json.loads(gzip.decompress((Path(run)/'REPLAYS.json.gz').read_bytes()))
ALIAS={'08fca639649c':'P1','7d161591d433':'P2','18c72e613b22':'P3','0622b2f8d354':'P4','89a4e072655d':'P5'}
def alias(k): return ALIAS.get(k[:12],k)
