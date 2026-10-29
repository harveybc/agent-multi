import json, sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs, evaluate, BASE, HOUR
from app.close_hold_controller import CloseController, CloseConfig, PathReport, value_paths
run=sys.argv[2]; front=json.loads((Path(run)/'FRONTIER.json').read_text()); G={r['key'][:12]:r['genes'] for r in front['rows']}
out={}
w=World(panel_specs(43)[0],CFG); e=w.env
R=replays(run); P1=[k for k in R if k.startswith('08fca639649c')][0]
tr=R[P1]['confirmation']['worlds'][0]['ledger'][0]; k=tr['entry_bar']; d=tr['direction']; b=k+3
s=.0003; E=e.o[k]+d*s/2
pos=type('P',(),dict(direction=d,entry_bar=k,entry_price=E))()
unit=w.position(tr['horizon'],d,E,k); pos.sl,pos.tp=unit.sl,unit.tp
fake=type('E',(),{'pos':pos,'v':e.v})()
u,vals,fo,fe=w.close_values(fake,tr['horizon'],b); r=w.report(b)
now=int(e.v[b])+HOUR
def attempt(f):
    try: return ('ACCEPTED',f())
    except Exception as x: return ('REFUSED',str(x))
c=lambda: CloseController(CloseConfig(.1,0.,1,.25),allow_synthetic=True)
out['full_route_current']=attempt(lambda: c().decide(u,r,now,fo,fe)['action'])
rs=w.report(b-1); ts=rs.first_fill+np.arange(rs.paths.shape[1])*HOUR
fos=w.funding(int(e.v[k]),d,ts); fes=w.funding(int(e.v[k]),d,ts+HOUR)
out['full_route_stale_report_at_env_clock']=attempt(lambda: c().decide(u,rs,now,fos,fes)['action'])
out['full_route_stale_report_at_report_clock(panel_convention)']=attempt(lambda: c().decide(u,rs,int(rs.issued_at),fos,fes)['action'])
rf=w.report(b+1); tf=rf.first_fill+np.arange(rf.paths.shape[1])*HOUR
out['full_route_future_report_at_env_clock']=attempt(lambda: c().decide(u,rf,now,w.funding(int(e.v[k]),d,tf),w.funding(int(e.v[k]),d,tf+HOUR))['action'])
vs=value_paths(u,rs,fos,fes)
out['cache_adapter_stale_values_at_env_clock']=attempt(lambda: c().decide_synthetic_values(u.identity,now,HOUR,vs,rs.weights,u.risk)['action'])
out['cache_adapter_forbidden_for_causal']=attempt(lambda: CloseController(allow_synthetic=False).decide_synthetic_values(u.identity,now,HOUR,vals,r.weights,u.risk))
out['full_route_synthetic_report_forbidden_for_causal']=attempt(lambda: CloseController(allow_synthetic=False).decide(u,r,now,fo,fe))
# F10: truncated self-consistent report (deadline moved earlier)
Hs=3; rt=replace(r,paths=r.paths[:,:Hs],spreads=r.spreads[:,:Hs],deadline=r.first_fill+Hs*HOUR)
out['F10_truncated_report_accepted']=attempt(lambda: c().decide(u,rt,now,fo[:Hs],fe[:Hs])['action'])
# cache hit with changed entry: same key, different entry price -> silently served
pos2=type('P',(),dict(direction=d,entry_bar=k,entry_price=E+.0010,sl=unit.sl,tp=unit.tp))()
u2,vals2,_,_=w.close_values(type('E',(),{'pos':pos2,'v':e.v})(),tr['horizon'],b)
out['cache_hit_changed_entry_served_stale']=bool(np.array_equal(vals2,vals) and u2.entry==E)
w.cache.clear()
out['cache_miss_changed_entry']=attempt(lambda: w.close_values(type('E',(),{'pos':pos2,'v':e.v})(),tr['horizon'],b) and 'built')
# changed side / profile / funding via the full route produce different valuations
up=w.position(6 if tr['horizon']!=6 else 3,d,E,k)
out['changed_profile_changes_values']=not np.allclose(value_paths(up,r,fo,fe),vals)
uo=w.position(tr['horizon'],-d,E,k)
fo2=w.funding(int(e.v[k]),-d,r.first_fill+np.arange(r.paths.shape[1])*HOUR); fe2=w.funding(int(e.v[k]),-d,r.first_fill+np.arange(r.paths.shape[1])*HOUR+HOUR)
out['changed_side_changes_values']=not np.allclose(value_paths(uo,r,fo2,fe2),vals)
out['changed_funding_changes_values']=not np.allclose(value_paths(u,r,fo+1e-4,fe+1e-4),vals)
# stale report injected into the whole panel: does verify=True catch it?
class Stale(World):
    def report(self,b): return super().report(max(self.env.first,b-1))
ws=Stale(panel_specs(43)[0],CFG)
out['panel_with_stale_reports_verify_true']=attempt(lambda: ws.run(G['08fca639649c'],verify=True)['return_pct'])
out['panel_true']=World(panel_specs(43)[0],CFG).run(G['08fca639649c'],verify=True)['return_pct']
# hidden labels: changing spec pattern/volatility labels after construction does not change decisions
w3=World(panel_specs(43)[0],CFG); base=w3.run(G['08fca639649c'],details=True)['trace']
w3.reports.clear(); w3.bids.clear(); w3.cache.clear(); w3.spec=dict(w3.spec,pattern='overlap',volatility='decreasing')
out['label_mutation_pattern_volatility_trace_equal']=w3.run(G['08fca639649c'],details=True)['trace']==base
w3.reports.clear(); w3.bids.clear(); w3.cache.clear(); w3.spec=dict(w3.spec,mirror=-1)
out['label_mutation_mirror_trace_equal']=w3.run(G['08fca639649c'],details=True)['trace']==base
print(json.dumps(out,indent=1,default=str))
