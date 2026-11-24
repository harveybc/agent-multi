"""Paired exit attribution at original notionals, not a new entry policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role461_seasonal_account import DEFAULT, NS, cost_arrays, exit_signal, load
from tools.role470_barriers import probe


def compare_position(i,side,q,h,capital,tm,sm,data,costs,signals,policy):
    t,o,_,_,c,atr,_,_,_,flat=data
    prof,spread,roll=costs
    width=atr[i-1]*np.sqrt(h)
    base=probe(i,side,tm*width,sm*width,q,h,data,costs)
    market=prof['slippage_market']
    en=o[i]+(spread[i] if side>0 else 0)+side*market
    sl=en-side*sm*width;tp=en+side*tm*width
    units=capital*.1*50*q/en
    passive=(base['end'],base['reason'],base['usd']*capital/10000)
    choices={'hold':passive,'oracle_bound':passive}
    ticks=[]
    for j in range(i+1,base['end']+1):
        # At open: calendar/gap has priority. Intrabar SL/TP has not happened yet.
        op=o[j]+(spread[j] if side<0 else 0.)
        if t[j]>=flat[i] or side*(op-sl)<=0 or side*(op-tp)>=0:break
        if t[j]-t[j-1]!=NS or not np.isfinite(signals[j-1]).all():continue
        carry=(roll[j]-roll[i])*prof['swap_long_per_lot' if side>0 else 'swap_short_per_lot']/prof['lot_units']
        net=(side*(op-en)-market-2*prof['commission_per_lot_side']/prof['lot_units']+carry)*units
        cur=c[j-1]+(spread[j-1] if side<0 else 0.)
        trigger=exit_signal(side,cur,sl,tp,signals[j-1],atr[j-1],policy)
        adverse=max(0.,float(np.max(-side*signals[j-1])))
        stop_distance=side*(cur-sl)
        ticks.append(dict(tick_ns=int(t[j]),net_if_exit=net,minus_hold=net-passive[2],trigger=trigger or 'hold',
                          held_hours=(t[j]-t[i])/NS,stop_distance=stop_distance,adverse=adverse,
                          projection_reaches_sl=adverse>=stop_distance))
        choices.setdefault('clock_first',(j,'clock',net))
        if 2*(t[j]-t[i])>=flat[i]-t[i]:
            choices.setdefault('clock_half',(j,'clock',net))
        if trigger=='early_sl':
            choices.setdefault('seasonal_sl',(j,trigger,net))
        if trigger in ('early_sl','early_pivot'):
            choices.setdefault('seasonal_sl_pivot',(j,trigger,net))
        if net>choices['oracle_bound'][2]:
            choices['oracle_bound']=(j,'oracle',net)
    for arm in ('clock_first','clock_half','seasonal_sl','seasonal_sl_pivot'):
        choices.setdefault(arm,passive)
    return choices,ticks


def main(a):
    cfg=json.loads((a.prev/'locked.json').read_text())['context']['effective']
    policy=json.loads(a.policy.read_text())['policy']
    data=load(a.src,a.bars,DEFAULT,a.prefix)
    t=data[0]
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    costs=(prof,*cost_arrays(t,json.loads((a.prefix/'CLOSURES.json').read_text()),prof))
    pred=np.load(a.signals);ts=pred['ts'];assert (np.diff(ts)>0).all() and ts.max()<pd.Timestamp('2024',tz='UTC').value
    ix=np.searchsorted(t+NS,ts);np.testing.assert_array_equal(t[ix]+NS,ts)
    assert np.isfinite(pred['x']).all()
    signals=np.full((len(t),8),np.nan);signals[ix]=pred['x']
    book=pd.read_csv(a.prev/'accounts.csv.gz').query('selector=="context" and arm=="deap"')
    records=[];actions=[]
    for r in book.itertuples():
        i=int(np.searchsorted(t,r.entry_ns));side=1 if r.side=='buy' else -1
        assert t[i]==r.entry_ns
        capital=r.balance-r.net_usd
        choices,ticks=compare_position(i,side,r.q,r.hours,capital,cfg['tp_atr_sqrt_h'],
                                        cfg['sl_atr_sqrt_h'],data,costs,signals,policy)
        end,reason,hold=choices['hold']
        assert t[end]==r.exit_bar_open_ns
        np.testing.assert_allclose(hold,r.net_usd,rtol=0,atol=1e-8)
        peak=choices['oracle_bound'][2]
        for arm,(j,why,value) in choices.items():
            assert value<=peak+1e-8
            records.append(dict(fold=r.fold,entry_ns=r.entry_ns,side=r.side,h=r.hours,q=r.q,
                original_capital=capital,hold_reason=reason,arm=arm,reason=why,exit_ns=t[j],
                opportunities=len(ticks),early=why in ('early_sl','early_pivot','clock','oracle'),
                usd=value,delta_usd=value-hold,regret_usd=peak-value,hours_held=(t[j]-t[i])/NS))
        for tick in ticks:
            actions.append(dict(fold=r.fold,entry_ns=r.entry_ns,side=r.side,**tick))
    df=pd.DataFrame(records);a.out.mkdir(exist_ok=True,parents=True)
    df.to_csv(a.out/'positions.csv.gz',index=False)
    pd.DataFrame(actions).to_csv(a.out/'alternatives.csv.gz',index=False)
    summary=[];pairs=[]
    for (fold,arm),g in df.groupby(['fold','arm']):
        summary.append(dict(fold=fold,arm=arm,trades=len(g),ret_fixed_pct=g.usd.sum()/100,
            delta_usd=g.delta_usd.sum(),closes=int(g.early.sum()),helpful=int((g.delta_usd>1e-8).sum()),
            harmful=int((g.delta_usd < -1e-8).sum()),wins=int((g.usd>0).sum()),regret_usd=g.regret_usd.sum()))
    for fold,g in df.groupby('fold'):
        panel=g.pivot(index='entry_ns',columns='arm',values='usd')
        week=pd.to_datetime(panel.index,utc=True).tz_localize(None).to_period('W-SUN')
        for arm in ('seasonal_sl','seasonal_sl_pivot'):
            for control in ('hold','clock_first','clock_half'):
                delta=(panel[arm]-panel[control]).groupby(week).agg(['sum','count'])
                draws=np.random.default_rng(473).integers(len(delta),size=(10000,len(delta)))
                samples=delta['sum'].to_numpy()[draws].sum(1)/delta['count'].to_numpy()[draws].sum(1)
                pairs.append(dict(fold=fold,arm=arm,control=control,weeks=len(delta),n=len(panel),
                    mean_usd=delta['sum'].sum()/len(panel),lo=np.quantile(samples,.025),hi=np.quantile(samples,.975)))
    pd.DataFrame(summary).to_csv(a.out/'summary.csv',index=False)
    pd.DataFrame(pairs).to_csv(a.out/'contrasts.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('src','bars','prefix','signals','prev','policy','out'):p.add_argument(k,type=Path)
    main(p.parse_args())
