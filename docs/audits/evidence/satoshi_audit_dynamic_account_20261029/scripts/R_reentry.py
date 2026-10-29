"""Re-entry / rotation analysis from REPLAYS ledgers only (own code)."""
import json, sys, gzip
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
R=json.loads(gzip.decompress((Path(sys.argv[1])/'REPLAYS.json.gz').read_bytes()))
A={'08fca639649c':'P1','7d161591d433':'P2','18c72e613b22':'P3','0622b2f8d354':'P4','89a4e072655d':'P5'}
out={}; tot=Counter(); last_hour=[]
for name,splits in R.items():
    al=A.get(name[:12],name)
    for split,res in splits.items():
        c=Counter(); gaps=Counter(); net_total=0.; net_after_early1=0.; net_after_early_any=0.; n_after_early1=0
        for w in res['worlds']:
            L=w['ledger']
            for i,t in enumerate(L):
                net_total+=t['net']
                if t['entry_bar']==107: last_hour.append(dict(policy=al,split=split,spec=w['spec'],entry=t['entry_bar'],exit=t['exit_bar'],reason=t['reason'],net=t['net'],horizon=t['horizon'],side=t['direction']))
                if i==0: c['first_of_week']+=1; continue
                p=L[i-1]; gap=t['entry_bar']-p['exit_bar']
                same='same' if t['direction']==p['direction'] else 'opposite'
                prev=p['reason']
                gaps[min(gap,10)]+=1
                c[f'gap{"1" if gap==1 else "2-3" if gap<=3 else "4-10" if gap<=10 else ">10"}|{prev}|{same}']+=1
                if gap==1: c['reentry_gap1']+=1
                if prev=='AGENT_CLOSE':
                    net_after_early_any+=t['net']
                    if gap==1: net_after_early1+=t['net']; n_after_early1+=1
        tot['trades']+=sum(len(w['ledger']) for w in res['worlds']); tot['gap1']+=c['reentry_gap1']
        out[f'{al}|{split}']=dict(counts=dict(sorted(c.items())),gap_hist={str(k):v for k,v in sorted(gaps.items())},
            net_total_usd=net_total,net_trades_opened_gap1_after_early=net_after_early1,n_trades_gap1_after_early=n_after_early1,
            share_of_net_gap1_after_early=net_after_early1/net_total if net_total else None,
            return_pts_gap1_after_early=100*net_after_early1/10000/12,
            net_after_early_any_gap=net_after_early_any)
print(json.dumps(dict(totals=dict(tot),entries_on_bar107=last_hour,per_policy=out),indent=1))
