"""Arms declared in G_PROTOCOL_PROSPECTIVE.md (sha 9177a23d...). Own driver; tracked files untouched."""
import json, sys, copy
from dataclasses import replace
from collections import Counter
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *; from ownval import levels, first_passage
import app.dynamic_account_panel as P
from app.dynamic_account_panel import World, panel_specs, evaluate, BASE, HOUR, expected_duration
from app.close_hold_controller import value_paths, CloseController
from app.dynamic_account_arbiter import HORIZONS
from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv
run=sys.argv[2]; outp=sys.argv[3]
front=json.loads((Path(run)/'FRONTIER.json').read_text()); genes={r['key']:r['genes'] for r in front['rows']}
P1=[k for k in genes if k.startswith('08fca639649c')][0]
POL={'P1':(genes[P1],None),'only_3':(list(BASE),{(3,1),(3,-1)})}
class StressWorld(World):
    def __init__(self,spec,cfg,sign,entries):
        super().__init__(spec,cfg); self.sign=sign; self.entries=entries
    def candidate_bids(self,b):
        if not self.entries: return super().candidate_bids(b)
        if b not in self.bids:
            r=self.report(b);rows=[];entry_bar=b+1
            times=r.first_fill+np.arange(r.paths.shape[1])*HOUR
            for h in HORIZONS:
                for d in (1,-1):
                    entry=float(r.weights@(r.paths[:,0,0]+d*r.spreads[:,0]/2))
                    p=self.position(h,d,entry,entry_bar); D=d*(entry-p.sl)
                    x=r.paths.copy(); x[:,1:,:]+=self.sign*d*.5*D
                    rr=replace(r,paths=x)
                    fo=self.funding(int(self.env.v[entry_bar]),d,times); fe=self.funding(int(self.env.v[entry_bar]),d,times+HOUR)
                    values=value_paths(p,rr,fo,fe)[:,1:]
                    means=rr.weights@values;j=int(np.argmax(means))
                    unc=float(np.sqrt(rr.weights@((values[:,j]-means[j])**2)))
                    rows.append(dict(horizon=h,direction=d,gain_risk=float(means[j])/p.risk,uncertainty_risk=unc/p.risk,
                                     duration=expected_duration(p,rr,j+1)))
            self.bids[b]=rows
        return self.bids[b]
    def close_values(self,e,h,b):
        p=e.pos;key=(h,p.direction,p.entry_bar,b)
        if key not in self.cache:
            r=self.report(b);unit=self.position(h,p.direction,p.entry_price,p.entry_bar)
            D=p.direction*(p.entry_price-unit.sl); x=r.paths+self.sign*p.direction*.5*D
            rr=replace(r,paths=x)
            times=r.first_fill+np.arange(r.paths.shape[1])*HOUR
            fo=self.funding(int(e.v[p.entry_bar]),p.direction,times); fe=self.funding(int(e.v[p.entry_bar]),p.direction,times+HOUR)
            self.cache[key]=(unit,value_paths(unit,rr,fo,fe),fo,fe)
        return self.cache[key]
def passive_cf(w,tr):
    e=w.env;f=e.cfg.profile;d=tr['direction'];k=tr['entry_bar']
    s=.00005 if w.spec['volatility']=='decreasing' else .0003;E=e.o[k]+d*s/2
    ref=w.references[tr['horizon'],d]
    sl,tp=levels(d,E,ref['tp_pips']*1e-4,ref['mae_pips']*1e-4,CFG['baseline'])
    q=np.column_stack((e.o,e.h,e.l,e.c))[tr['exit_bar']:w.end]-d*s/2
    at,X,reason,at_open=first_passage(d,sl,tp,q,f.limit_through)
    reason='FORCED_FLAT' if reason=='DEADLINE' else reason
    rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
    jx=tr['exit_bar']+at; ex=int(e.v[jx])+(0 if at_open else HOUR)
    fin=sum(rate*tr['units']*m for t0,m in w.rolls if int(e.v[k])<t0<=ex)
    xs=f.slippage_stop if reason=='SL' else 0. if reason=='TP' else f.slippage_market
    return reason,d*tr['units']*(X-E)-tr['units']*(f.slippage_market+xs)-2*tr['units']/f.lot_units*f.commission_per_lot_side-fin
def summarize(worlds,res):
    c=Counter()
    for w,r in zip(worlds,res['worlds']):
        for t in r['ledger']:
            c['orders']+=1;c[t['reason']]+=1;c['wins']+=t['net']>0
            if t['reason']=='AGENT_CLOSE':
                pr,pn=passive_cf(w,t);c['early_passive_'+pr]+=1;c['early_destructive']+=t['net']<pn
    return dict(return_pct=res['objectives'][0],worst_dd_pct=-res['objectives'][1],eligible=res['eligible'],
                orders_per_week=res['mean_orders'],per_week_returns=[r['return_pct'] for r in res['worlds']],**dict(c))
class AlwaysHold(CloseController):
    def _decide_values(self,*a,**k):
        r=super()._decide_values(*a,**k); r['action']='HOLD'; return r
orig_gate=AbsoluteBarEnv.entry_gate
def cooldown(n):
    def gate(self,b):
        ok=orig_gate(self,b)
        if ok[0] and self.trades and self.trades[-1]['close_reason']=='AGENT_CLOSE' and b+1-self.trades[-1]['exit_bar_abs']<=n:
            return False,'AUDIT_COOLDOWN'
        return ok
    return gate
out={}
for split,seed in (('calibration',7),('confirmation',43)):
    for pol,(g,en) in POL.items():
        base=[World(s,CFG) for s in panel_specs(seed)]
        out[f'baseline|{pol}|{split}']=summarize(base,evaluate(base,g,en,details=True))
        for arm,sign,ent in (('G1',-1,True),('G1b',-1,False),('G2',+1,True)):
            ws=[StressWorld(s,CFG,sign,ent) for s in panel_specs(seed)]
            out[f'{arm}|{pol}|{split}']=summarize(ws,evaluate(ws,g,en,details=True))
        P.CloseController=AlwaysHold
        try: out[f'G3a_never_close|{pol}|{split}']=summarize(base,evaluate(base,g,en,details=True))
        finally: P.CloseController=CloseController
        for arm,n in (('G3b_cooldown1',1),('G3c_cooldown6',6)):
            AbsoluteBarEnv.entry_gate=cooldown(n)
            try:
                ws=[World(s,CFG) for s in panel_specs(seed)]
                out[f'{arm}|{pol}|{split}']=summarize(ws,evaluate(ws,g,en,details=True))
            finally: AbsoluteBarEnv.entry_gate=orig_gate
# G4 duplicate groups, confirmation only
rows={r['key']:r for r in front['rows']}
conf=[World(s,CFG) for s in panel_specs(43)]
for al,pref in (('P1','08fca639649c'),('P2','7d161591d433'),('P3','18c72e613b22')):
    k=[x for x in rows if x.startswith(pref)][0]; vec=rows[k]['objectives']
    grp=[x for x in rows if rows[x]['objectives']==vec]
    res=[evaluate(conf,rows[x]['genes'])['objectives'] for x in grp]
    r=np.array(res)
    out[f'G4|{al}']=dict(n=len(grp),conf_return_min=float(r[:,0].min()),conf_return_max=float(r[:,0].max()),
        conf_dd_min=float(-r[:,1].max()),conf_dd_max=float(-r[:,1].min()),distinct_conf_vectors=len({tuple(x) for x in res}),
        representative_conf=evaluate(conf,rows[k]['genes'])['objectives'])
Path(outp).write_text(json.dumps(out,indent=1,default=float))
