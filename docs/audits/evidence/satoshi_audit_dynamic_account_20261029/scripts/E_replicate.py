"""Independent replication of every entry and close decision in the 288 stored accounts,
plus cache-vs-actual-unit comparison, SL-foresight counts and passive counterfactuals of early closes."""
import json, math, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *; from ownval import *
from app.dynamic_account_panel import World, panel_specs
from app.dynamic_account_arbiter import decode
from app.close_hold_controller import CloseController, CloseConfig
run=sys.argv[2]; R=replays(run)
front=json.loads((Path(run)/'FRONTIER.json').read_text()); genes={r['key']:r['genes'] for r in front['rows']}
from tools.optimize_dynamic_account import controls
ctrl={n:(g,e) for n,g,e in controls()}
refs={(r['horizon'],1 if r['direction']=='buy' else -1):r['reference']['median'] for r in CFG['rows']}
bg=CFG['baseline']
panels={'calibration':[World(s,CFG) for s in panel_specs(7)],'confirmation':[World(s,CFG) for s in panel_specs(43)]}
C=Counter(); M=defaultdict(float); early_cf=Counter(); early_rows=[]
def fund_arr(w,entry_ns,d,times,pos_only=False):
    f=w.env.cfg.profile; rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
    if pos_only: rate=max(0.,rate)
    return np.array([sum(rate*m for t,m in w.rolls if entry_ns<t<=x) for x in times])
for name,splits in R.items():
    if name in ctrl: g,en=ctrl[name]
    else: g,en=genes[name],None
    g=decode(g)
    for split,res in splits.items():
        for w,rec in zip(panels[split],res['worlds']):
            e=w.env; f=e.cfg.profile
            prof=dict(slip_m=f.slippage_market,slip_s=f.slippage_stop,comm_rt=2*f.commission_per_lot_side/f.lot_units,through=f.limit_through)
            led=rec['ledger']; trace={t['bar']:t for t in rec['trace']}
            streak=0; last=None; ident=None; bid_cache={}
            for b in range(e.first,e.last):
                t=trace[b]
                pos=next((x for x in led if x['entry_bar']<=b<x['exit_bar']),None)
                if pos is not None:
                    C['occupied_ticks']+=1
                    if b+1>=w.end:
                        C['occupied_no_report']+=1; assert t['action']==0; continue
                    r=w.report(b); d=pos['direction']; k=pos['entry_bar']
                    assert r.issued_at==int(e.v[b])+H and r.deadline==int(e.ff[0]) and r.paths.shape[1]==w.end-b-1
                    E=e.o[k]+d*r.spreads[0,0]/2
                    sl,tp=levels(d,E,refs[pos['horizon'],d]['tp_pips']*1e-4,refs[pos['horizon'],d]['mae_pips']*1e-4,bg)
                    times=r.first_fill+np.arange(r.paths.shape[1])*H
                    fo=fund_arr(w,int(e.v[k]),d,times); fe=fund_arr(w,int(e.v[k]),d,times+H)
                    reserve=fund_arr(w,int(e.v[k]),d,[int(e.ff[0])],True)[0]
                    perunit=d*(E-sl)+prof['slip_m']+prof['slip_s']+prof['comm_rt']+reserve
                    vals,meta=values(d,E,sl,tp,pos['units'],r.paths,r.spreads,fo,fe,prof)
                    adv=advantage(vals,r.weights,g[5],perunit*pos['units'])
                    vals1,_=values(d,E,sl,tp,1.,r.paths,r.spreads,fo,fe,prof)
                    adv1=advantage(vals1,r.weights,g[5],perunit)
                    M['unit_vs_actual_adv']=max(M['unit_vs_actual_adv'],abs(adv-adv1))
                    # Musashi's cached per-unit valuation for the same key
                    unit,cv,_,_=w.close_values(type('E',(),{'pos':type('P',(),dict(direction=d,entry_bar=k,entry_price=E,sl=sl,tp=tp))(),'v':e.v})(),pos['horizon'],b)
                    advc=advantage(cv,r.weights,g[5],unit.risk)
                    M['own_vs_cache_adv']=max(M['own_vs_cache_adv'],abs(adv-advc))
                    if any(m[1]=='SL' for m in meta): C['occupied_ticks_some_sample_SL']+=1
                    if all(m[1]=='SL' for m in meta): C['occupied_ticks_all_samples_SL']+=1
                    idn=f"{pos['horizon']}:{d}:{k}"
                    tick=int(r.issued_at)
                    cont=ident==idn and last is not None and tick==last+H
                    s0=streak if cont else 0
                    q=adv>max(g[9],1e-12); streak=s0+1 if q else 0; ident=idn; last=tick
                    urgent=q and adv>=.25
                    close=q and (urgent or streak>=g[10])
                    if abs(adv-g[9])<1e-9: C['near_margin_ticks']+=1
                    if close!=(t['action']==3): C['close_mismatch']+=1
                    C['close_decisions']+=close
                    if close and k==b: C['entry_bar_close']+=1
                else:
                    ok=(b+1<=e.last and e.seg_of[b+1]==e.seg_of[b] and e.v[b+1]>=e.op[0] and e.v[b+1]<e.ff[0])
                    if not ok:
                        if t['action'] in (1,2): C['entry_outside_gate']+=1
                        continue
                    C['flat_gate_ticks']+=1
                    r=w.report(b)
                    if b not in bid_cache:
                        rows=[]
                        times=r.first_fill+np.arange(r.paths.shape[1])*H
                        for h in HZ:
                            for d in (1,-1):
                                E=float(r.weights@(r.paths[:,0,0]+d*r.spreads[:,0]/2))
                                sl,tp=levels(d,E,refs[h,d]['tp_pips']*1e-4,refs[h,d]['mae_pips']*1e-4,bg)
                                fo=fund_arr(w,int(e.v[b+1]),d,times); fe=fund_arr(w,int(e.v[b+1]),d,times+H)
                                reserve=fund_arr(w,int(e.v[b+1]),d,[int(e.ff[0])],True)[0]
                                perunit=d*(E-sl)+prof['slip_m']+prof['slip_s']+prof['comm_rt']+reserve
                                vals,meta=values(d,E,sl,tp,1.,r.paths,r.spreads,fo,fe,prof)
                                v=vals[:,1:]; mu=r.weights@v; j=int(np.argmax(mu))
                                unc=float(np.sqrt(r.weights@((v[:,j]-mu[j])**2)))
                                dur=float(r.weights@np.array([min(j+1,m[0] if m[2] else m[0]+1) for m in meta]))
                                rows.append(dict(h=h,d=d,gain_risk=float(mu[j])/perunit,unc=unc/perunit,dur=dur,
                                                 sl_any=any(m[1]=='SL' for m in meta)))
                        bid_cache[b]=rows
                    rows=bid_cache[b]
                    if en is not None: rows=[x for x in rows if (x['h'],x['d']) in en]
                    x=np.diff(e.c[max(0,b-6):b+1]);den=float(np.abs(x).sum());eff=float(x.sum()/den) if den else 0.
                    sel=choose(rows,g,eff)
                    exp=0 if sel is None else (1 if sel['d']==1 else 2)
                    if exp!=t['action'] or (sel and sel['h']!=t['horizon']): C['entry_mismatch']+=1
                    if sel: C['entries']+=1
                    streak=0
            # passive counterfactual of each early close (native rules, own simulation, true bars)
            s=.00005 if w.spec['volatility']=='decreasing' else .0003
            for tr in led:
                if tr['reason']!='AGENT_CLOSE': continue
                d=tr['direction'];k=tr['entry_bar'];E=e.o[k]+d*s/2
                sl,tp=levels(d,E,refs[tr['horizon'],d]['tp_pips']*1e-4,refs[tr['horizon'],d]['mae_pips']*1e-4,bg)
                q=np.column_stack((e.o,e.h,e.l,e.c))[tr['exit_bar']:w.end]-d*s/2
                at,X,reason,at_open=first_passage(d,sl,tp,q,f.limit_through)
                if reason=='DEADLINE': reason='FORCED_FLAT'
                rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
                jx=tr['exit_bar']+at; exit_ns=int(e.v[jx])+(0 if at_open else H)
                fin=sum(rate*tr['units']*m for t0,m in w.rolls if int(e.v[k])<t0<=exit_ns)
                xs=f.slippage_stop if reason=='SL' else 0. if reason=='TP' else f.slippage_market
                pnet=d*tr['units']*(X-E)-tr['units']*(f.slippage_market+xs)-2*tr['units']/f.lot_units*f.commission_per_lot_side-fin
                early_cf[(alias(name),split,reason)]+=1
                early_cf[(alias(name),split,'helpful' if tr['net']>pnet else 'destructive')]+=1
                if alias(name) in ('P1','only_3') and split=='confirmation':
                    early_rows.append(dict(p=alias(name),world=w.spec,entry=k,exit=tr['exit_bar'],net=tr['net'],passive=reason,passive_net=pnet))
print(json.dumps(dict(counts=dict(C),max_errors=dict(M),early_close_counterfactual={'|'.join(k):v for k,v in sorted(early_cf.items())},
     early_rows=early_rows),indent=1,default=float))
