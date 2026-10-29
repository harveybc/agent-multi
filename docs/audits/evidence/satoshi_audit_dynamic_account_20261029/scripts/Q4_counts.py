import json,sys
from collections import Counter
sys.path.insert(0,sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs
import numpy as np
R=replays(sys.argv[2]); panels={'calibration':[World(s,CFG) for s in panel_specs(7)],'confirmation':[World(s,CFG) for s in panel_specs(43)]}
c=Counter()
for name,sp in R.items():
    for split,res in sp.items():
        for w,rec in zip(panels[split],res['worlds']):
            led=rec['ledger']; c['agent_close_ledger']+=sum(t['reason']=='AGENT_CLOSE' for t in led)
            c['close_actions_trace']+=sum(t['action']==3 for t in rec['trace'])
            for t in rec['trace']:
                if t['action']==3:
                    tr=next(x for x in led if x['entry_bar']<=t['bar']<x['exit_bar'])
                    if tr['exit_bar']!=t['bar']+1: c['close_decision_not_executed_at_next_open']+=1
                    elif tr['reason']!='AGENT_CLOSE': c[f'close_preempted_by_{tr["reason"]}_at_open']+=1
            e=w.env; gap=np.flatnonzero(np.abs(e.o[1:]-e.c[:-1])>5e-4)+1
            for t in led:
                if any(t['entry_bar']<g<=t['exit_bar'] for g in gap): c['trades_spanning_gap']+=1; c[f'gap_span_exit_{t["reason"]}']+=1
            c['worlds_with_gap']+=len(gap)>0
print(json.dumps(dict(c),indent=1))
