"""Re-evaluate EVERY receipt from scratch (reverse order, so cache state differs from the search)."""
import json, sys, time
from pathlib import Path
am, hs, run, outp = map(Path, sys.argv[1:5])
sys.path.insert(0, str(am))
from app.zigzag_opportunities import engine
engine(hs)
from app.dynamic_account_panel import World, panel_specs, evaluate
from tools.optimize_dynamic_account import compact
cfg=json.loads((am/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
worlds=[World(s,cfg) for s in panel_specs(7)]
rows=[json.loads(l) for l in (run/'CANDIDATES.jsonl').read_text().splitlines()]
bad=[]; t=time.time()
for i,r in enumerate(reversed(rows)):
    got=compact(evaluate(worlds,r['genes']))
    if got!=r['result']: bad.append(dict(key=r['key'],stored=r['result'],got=got))
    if i%1000==0: print(i,len(bad),round(time.time()-t,1),flush=True)
outp.write_text(json.dumps(dict(evaluated=len(rows),mismatches=len(bad),first=bad[:5],seconds=time.time()-t),indent=1))
print('DONE',len(rows),len(bad))
