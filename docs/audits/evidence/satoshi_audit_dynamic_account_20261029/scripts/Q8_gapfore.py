import json,sys
from collections import Counter
sys.path.insert(0,sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs
import numpy as np
R=replays(sys.argv[2]); panels={'calibration':[World(s,CFG) for s in panel_specs(7)],'confirmation':[World(s,CFG) for s in panel_specs(43)]}
c=Counter(); fam=Counter()
for split,ws in panels.items():
    for w in ws:
        e=w.env; g=np.flatnonzero(np.abs(e.o[1:]-e.c[:-1])>5e-4)+1
        fam[(split,w.spec['pattern'],'gap_bar',int(g[0]) if len(g) else None)]+=1
for name,sp in R.items():
    for split,res in sp.items():
        for w,rec in zip(panels[split],res['worlds']):
            e=w.env; g=np.flatnonzero(np.abs(e.o[1:]-e.c[:-1])>5e-4)+1
            if not len(g): continue
            g=int(g[0])
            for t in rec['ledger']:
                if t['reason']=='AGENT_CLOSE' and t['entry_bar']<g:
                    dist=g-t['exit_bar']
                    if 0<=dist<=3: c[(ALIAS.get(name[:12],name),split,'early_close_0to3_bars_before_gap')]+=1
                    if dist==0: c[(ALIAS.get(name[:12],name),split,'early_close_at_last_open_before_gap')]+=0
                c[(ALIAS.get(name[:12],name),split,'early_total_gapworld')]+=t['reason']=='AGENT_CLOSE'
print(json.dumps({'|'.join(map(str,k)):v for k,v in sorted(c.items(),key=str)},indent=0))
print(json.dumps({'|'.join(map(str,k)):v for k,v in sorted(fam.items(),key=str)},indent=0))
