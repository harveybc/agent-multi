"""Refit just the discretionary SL trigger; paired shadow exits reuse original fills."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tools.role461_seasonal_account import DEFAULT,NS,exit_signal,hit,simulate
from tools.role464_deap import prepare
from tools.role465_accuracy_refit import guarded_hit


def fixed(row,market,params,discretionary):
    data,windows,prof,spread,roll=market
    t,o,hi,lo,c,_,ds,_,_,deadline=data
    s=row['side'];en=row['entry_price'];start=row['entry_index'];sl=row['sl'];tp=row['tp']
    for j in range(start,len(t)):
        add=spread[j] if s<0 else 0.;px=o[j]+add
        if t[j]>=deadline[start]:
            assert t[j]==deadline[start]
            reason='calendar'
        elif s*(px-sl)<=0:reason='sl_gap'
        elif s*(px-tp)>=0:reason='tp_gap'
        elif (discretionary and j>start and t[j]-t[j-1]==NS and
              exit_signal(s,c[j-1]+(spread[j-1] if s<0 else 0.),sl,tp,ds[j-1],
                          windows[params['atr_window']][j-1],params)=='early_sl'):
            reason='early_sl'
        else:
            barrier=hit(s,px,hi[j]+add,lo[j]+add,sl,tp,prof['limit_through'])
            if not barrier:continue
            reason,px=barrier
        slip=prof['slippage_stop'] if reason.startswith('sl') else (
            0. if reason.startswith('tp') else prof['slippage_market'])
        swap=prof['swap_long_per_lot' if s>0 else 'swap_short_per_lot']
        pnl=row['units']*(s*(px-s*slip-en)+((roll[j]-roll[start])*swap-
                            2*prof['commission_per_lot_side'])/prof['lot_units'])
        return j,reason,pnl
    raise ValueError('position has no terminal exit')


def main(a):
    cfg=json.loads((a.prev/'config.json').read_text())
    orig=json.loads((a.prev/'locked.json').read_text())['cost']['effective']
    cfg['genes']['early_margin_atr']={}
    par=tuple(orig.values());ev,ref,_,market=prepare(a,cfg,with_market=True)
    years=cfg['train_years'];vy=cfg['validation_year'];ty=cfg['report_year']
    base={y:ev((*par,0.),y,True) for y in years+[vy]}
    archive=json.loads((a.prev/'results.json').read_text())['panels']['cost']['cost']
    grid=[];annual=[]
    for m in np.arange(-40,41)/20:
        rows=[ev((*par,m),y,True) for y in years]
        annual.extend(dict(year=y,margin=m,**v) for y,v in zip(years,rows))
        score=guarded_hit(rows,[base[y] for y in years])
        rate=sum(v['wins'] for v in rows)/sum(v['orders'] for v in rows)
        grid.append(dict(margin=m,score=score,win_rate=rate))
    lead=max(grid,key=lambda r:(r['score'],-abs(r['margin'])))
    m=lead['margin'];v=ev((*par,m),vy,True)
    adopted=(lead['score']>guarded_hit([base[y] for y in years],[base[y] for y in years])+1e-8
             and guarded_hit([v],[base[vy]])>guarded_hit([base[vy]],[base[vy]])+1e-8)
    lock=dict(train_leader=m,adopted=bool(adopted),selected=m if adopted else 0.,
              diagnostics=[-.5,0.,.5],train_feasible=int(sum(r['score']>=0 for r in grid)))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'locked.json').write_text(json.dumps(lock,indent=2)+'\n')
    margins=sorted(set(lock['diagnostics']+[m]))
    audit=[];matched=[];no_early=[]
    for y in years+[vy,ty]:
        v0,book=ref((*par,0.),y,True,with_book=True)
        data,windows,prof,spread,roll=market;shadow_data=data.copy();shadow_data[5]=windows[orig['atr_window']]
        hold,_=simulate(shadow_data,(None,None),dict(DEFAULT,**orig),y,'bridge',False,(prof,spread,roll))
        no_early.append(dict(year=y,**{k:hold[k] for k in ('return_pct','dd_close_pct','trades','buy','wins','wins_buy')}))
        for k,z in ev((*par,0.),y,True).items():assert np.isclose(z,archive[str(y)][k],atol=1e-8,rtol=0)
        for margin in margins:
            if margin or y in (vy,ty):
                _,vbook=ref((*par,margin),y,True,with_book=True)
            else:vbook=book
            actual=ev((*par,margin),y,True)
            audit.append(dict(year=y,margin=margin,**actual,
                              buy_wins=sum(r['pnl']>0 and r['side']==1 for r in vbook)))
            for r in book:
                j,reason,pnl=fixed(r,market,dict(DEFAULT,**orig,early_margin_atr=margin),True)
                _,_,hold=fixed(r,market,dict(DEFAULT,**orig),False)
                if margin==0:
                    assert (j,reason)==(r['exit_index'],r['reason'])
                    np.testing.assert_allclose(pnl,r['pnl'],rtol=0,atol=1e-8)
                    np.testing.assert_allclose(pnl-hold,r.get('early_minus_hold',0.),rtol=0,atol=1e-8)
                matched.append(dict(year=y,margin=margin,entry=r['entry_index'],side=r['side'],
                                    incumbent_exit=r['exit_index'],exit=j,reason=reason,
                                    incumbent_net=r['pnl'],net=pnl,hold_net=hold))
    for name,records in [('train_grid.csv',grid),('train_years.csv',annual),('accounts.csv',audit),
                         ('fixed_entries.csv.gz',matched),('no_early_control.csv',no_early)]:
        pd.DataFrame(records).to_csv(a.out/name,index=False)
    pairs=pd.DataFrame(matched)
    for tag,col in [('vs_original','incumbent_net'),('vs_hold','hold_net')]:
        delta=pairs.net-pairs[col]
        pairs[tag+'_usd']=delta
        pairs[tag+'_improved']=delta>1e-7
        pairs[tag+'_worsened']=delta< -1e-7
        pairs[tag+'_loss_to_win']=(pairs[col]<0)&(pairs.net>0)
        pairs[tag+'_win_to_loss']=(pairs[col]>0)&(pairs.net<0)
    cols=[c for c in pairs if c.startswith('vs_')]
    pairs.groupby(['year','margin','side'])[cols].sum().to_csv(a.out/'paired_summary.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('src','bars','prefix','recipe','prev','out'):p.add_argument(k,type=Path)
    main(p.parse_args())
