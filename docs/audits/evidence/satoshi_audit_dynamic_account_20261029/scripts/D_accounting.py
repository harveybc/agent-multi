"""Own cash / sizing / exit / MTM-drawdown / gate recomputation from REPLAYS ledgers.
Uses the World object ONLY for exogenous bars, calendar and cost constants; every
price, level, size, exit and cash number below is computed here."""
import json, math, sys
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs
run=sys.argv[2]; R=replays(run)
H=3600*10**9
def levels(d,e,tp_ref,mae_ref,g,tick=1e-5):
    D=lambda x: Decimal(str(x))
    ds=D(g['a_sl'])*D(mae_ref)+D(g['b_sl'])*D(.001)+D(g['buffer_pips'])*D(.0001)
    dt=D(g['a_tp'])*D(tp_ref)-D(g['b_tp'])*D(.001)
    rnd=ROUND_FLOOR if d==1 else ROUND_CEILING
    r=lambda x: float((x/D(tick)).to_integral_value(rounding=rnd)*D(tick))
    return r(D(e)-D(d)*ds), r(D(e)+D(d)*dt)
refs={(r['horizon'],1 if r['direction']=='buy' else -1):r['reference']['median'] for r in CFG['rows']}
g=CFG['baseline']
panels={'calibration':[World(s,CFG) for s in panel_specs(7)],'confirmation':[World(s,CFG) for s in panel_specs(43)]}
tot=Counter(); errs=defaultdict(float); rows_out=[]; exit_kinds=Counter(); per_policy={}
F07=Counter()
for name,splits in R.items():
    per_policy[alias(name)]={}
    for split,res in splits.items():
        agg=Counter(); rets=[]; dds=[]; sharpes=[]
        for w,rec in zip(panels[split],res['worlds']):
            assert w.spec==rec['spec']
            e=w.env; f=e.cfg.profile
            s=.00005 if w.spec['volatility']=='decreasing' else .0003
            assert all(f.spread_at(int(t),int(e.mo[0]))==s for t in e.v) and f.spread_at(int(e.v[-1])+H,int(e.mo[0]))==s
            rolls=[(int(t.value),3 if i==3 else 1) for i,t in enumerate(w.env.cal.segments[0].rollovers(),1)]
            op=int(e.op[0]); ff=int(e.ff[0])
            bal=10000.; prev_exit=-1; equity_pts=[]  # (bar, equity at close of bar)
            pos_by_bar={}
            for t in rec['ledger']:
                d=t['direction']; k=t['entry_bar']; j_led=t['exit_bar']; u=t['units']
                tot['trades']+=1
                # gates
                if not (op<=e.v[k]<ff): tot['gate_violation']+=1
                if k<=prev_exit: tot['overlap_or_same_tick_reopen']+=1
                if k==prev_exit+1: tot['reentry_next_bar']+=1
                if e.v[k]+H>=ff: tot['entry_on_forced_flat_bar']+=1
                E=e.o[k]+d*s/2
                ref=refs[t['horizon'],d]
                sl,tp=levels(d,E,ref['tp_pips']*1e-4,ref['mae_pips']*1e-4,g)
                # sizing
                reserve=sum(max(0.,-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units)*m for tt,m in rolls if int(e.v[k])<tt<=ff)
                umax=math.floor(bal*.1*50/E)
                lpu=d*(E-sl)+f.slippage_market+f.slippage_stop+2*f.commission_per_lot_side/f.lot_units+reserve
                qual=None
                # quality (size gene) is recovered from q and the risk ratio
                ratio=min(1.,g['rho']*bal/(umax*lpu))
                qual=t['q']/ratio
                units=math.floor(t['q']*umax)
                if units!=u: tot['units_mismatch']+=1
                margin=u*E/50
                if margin>t['q']*.1*bal+1e-9: tot['margin_over_q01B']+=1
                if margin>.1*bal+1e-9: tot['margin_over_01B']+=1
                if u*lpu>g['rho']*bal*qual+1e-6: tot['risk_over_budget']+=1
                errs['q_quality_dev_from_grid']=max(errs['q_quality_dev_from_grid'],abs(round(qual,3)-qual))
                # exit simulation
                reason=None
                for j in range(k,len(e.o)):
                    oq=e.o[j]-d*s/2
                    if j>k:
                        if d*(oq-sl)<=0: reason,X,kind,j_ex='SL',oq,'SL_OPEN',j;break
                        if d*(oq-tp)>=0: reason,X,kind,j_ex='TP',oq,'TP_OPEN',j;break
                        if t['reason']=='AGENT_CLOSE' and j==j_led: reason,X,kind,j_ex='AGENT_CLOSE',oq,'AGENT_OPEN',j;break
                    hq=e.h[j]-d*s/2; lq=e.l[j]-d*s/2
                    adverse=lq if d==1 else hq; fav=hq if d==1 else lq
                    native_tp=(hq>=tp+f.limit_through) if d==1 else (lq<=tp-f.limit_through)
                    ref_tp=d*(fav-tp)>=f.limit_through
                    F07['bars_checked']+=1
                    if native_tp!=ref_tp: F07['native_vs_reference_disagree']+=1
                    if abs(d*(fav-tp)-f.limit_through)<1e-7: F07['within_1e-7_of_boundary']+=1
                    if d*(adverse-sl)<=0: reason,X,kind,j_ex='SL',sl,'SL_INTRABAR',j;break
                    if native_tp: reason,X,kind,j_ex='TP',tp,'TP_INTRABAR',j;break
                    if e.v[j]+H>=ff: reason,X,kind,j_ex='FORCED_FLAT',e.c[j]-d*s/2,'FORCED_FLAT',j;break
                exit_kinds[(alias(name),split,kind)]+=1
                if reason!=t['reason'] or j_ex!=j_led: tot['exit_mismatch']+=1
                if reason=='AGENT_CLOSE' and j_ex==k: tot['entry_bar_voluntary_close']+=1
                at_open=kind.endswith('OPEN')
                exit_ns=int(e.v[j_ex])+(0 if at_open else H)
                rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
                fin=sum(rate*u*m for tt,m in rolls if int(e.v[k])<tt<=exit_ns)
                xslip={'SL':f.slippage_stop,'TP':0.}.get(reason,f.slippage_market)
                net=d*u*(X-E)-u*(f.slippage_market+xslip)-2*u/f.lot_units*f.commission_per_lot_side-fin
                errs['net_abs']=max(errs['net_abs'],abs(net-t['net']))
                agg['spread']+=u*s; agg['slip']+=u*(f.slippage_market+xslip); agg['comm']+=2*u/f.lot_units*f.commission_per_lot_side; agg['fin']+=fin
                for b in range(k,j_ex+1): pos_by_bar[b]=(d,u,E,k,j_ex,at_open,bal)
                bal+=net; prev_exit=j_ex
                agg['n']+=1; agg['win']+=net>0; agg[reason]+=1; agg['win_'+reason]+=net>0; agg['loss_'+reason]+=net<=0
                agg['buy']+=d==1; agg['buywin']+=(d==1 and net>0); agg['sellwin']+=(d==-1 and net>0)
            # MTM equity at end of each processed bar k=1..last
            eq=[10000.]; b2=10000.; cash_after={}
            run_bal=10000.
            led=rec['ledger']; idx=0; balance=10000.
            closes_at={t['exit_bar']:t for t in led}
            bal_t=10000.
            for kbar in range(1,e.last+1):
                # realize exits at this bar
                if kbar in closes_at: bal_t+=closes_at[kbar]['net']
                pb=pos_by_bar.get(kbar)
                if pb and pb[4]!=kbar:
                    d,u,E,k0,j0,_,_=pb
                    mark=e.c[kbar]-d*s/2
                    rate=-(f.swap_long_per_lot if d==1 else f.swap_short_per_lot)/f.lot_units
                    fin=sum(rate*u*m for tt,m in rolls if int(e.v[k0])<tt<=int(e.v[kbar])+H)
                    eq.append(bal_t+d*u*(mark-E)-u*f.slippage_market-u/f.lot_units*f.commission_per_lot_side-fin)
                else: eq.append(bal_t)
            x=np.r_[10000.,np.array(eq)]; peak=np.maximum.accumulate(x)
            ret=100*(x[-1]/10000-1); dd=100*np.max((peak-x)/peak)
            errs['return_pct']=max(errs['return_pct'],abs(ret-rec['return_pct'])); errs['dd_pct']=max(errs['dd_pct'],abs(dd-rec['drawdown_pct']))
            # closed-trade-only drawdown for contrast
            cl=np.r_[10000.,10000.+np.cumsum([t['net'] for t in led])]; pk=np.maximum.accumulate(cl)
            agg['closed_dd_max']=max(agg['closed_dd_max'],100*np.max((pk-cl)/pk))
            # daily Sharpe, own: equity at each bar close, group by UTC day
            import pandas as pd
            ts=pd.DatetimeIndex(e.v[1:e.last+1]+H,tz='UTC') if False else pd.to_datetime(e.v[1:e.last+1]+H,utc=True)
            ser=pd.Series(eq[1:],index=ts)
            # native uses bars first..last with first bar equity == capital minus entry cost; bar 0 equity is capital
            ser=pd.concat([pd.Series([10000.],index=pd.to_datetime([e.v[0]+H],utc=True)),ser])
            day=ser.groupby(ser.index.normalize()).last()
            r=np.diff(np.r_[10000.,day.to_numpy()])/np.r_[10000.,day.to_numpy()][:-1]
            sh=r.mean()/r.std(ddof=1)*math.sqrt(365)
            errs['sharpe']=max(errs['sharpe'],abs(sh-rec['native_sharpe']))
            agg['days']=len(r)
            rets.append(rec['return_pct']); dds.append(rec['drawdown_pct']); sharpes.append(rec['native_sharpe'])
        n=agg['n']
        weekly=np.array(rets)
        per_policy[alias(name)][split]=dict(mean_return=float(weekly.mean()),worst_dd=float(max(dds)),orders=n,orders_week=n/12,
            buys=agg['buy'],sells=n-agg['buy'],win_pct=100*agg['win']/n if n else None,buy_wins=agg['buywin'],sell_wins=agg['sellwin'],
            TP=agg['TP'],SL=agg['SL'],FF=agg['FORCED_FLAT'],EARLY=agg['AGENT_CLOSE'],
            win_by_reason={r:(agg['win_'+r],agg['loss_'+r]) for r in ('TP','SL','AGENT_CLOSE','FORCED_FLAT')},
            mean_sharpe=float(np.mean(sharpes)),weekly_sd=float(weekly.std(ddof=1)),
            weekly_sharpe_sqrt52=float(weekly.mean()/weekly.std(ddof=1)*math.sqrt(52)) if weekly.std(ddof=1)>0 else None,
            costs=dict(spread=agg['spread'],slip=agg['slip'],comm=agg['comm'],fin=agg['fin']),closed_trade_dd_max=agg['closed_dd_max'],
            objectives_match=abs(weekly.mean()-res['objectives'][0])<1e-12 and abs(-max(dds)-res['objectives'][1])<1e-12,days=agg['days'])
out=dict(totals=dict(tot),max_abs_errors=dict(errs),F07_boundary=dict(F07),
         exit_kinds={f'{a}|{b}|{c}':v for (a,b,c),v in sorted(exit_kinds.items())},per_policy=per_policy)
print(json.dumps(out,indent=1,default=float))
