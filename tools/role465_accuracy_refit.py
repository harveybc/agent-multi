"""Warm DEAP on ideal forecasts with fees, without redefining the strategy."""
import argparse
import json
from pathlib import Path
import numpy as np
from deap import base,creator
from tools.role464_deap import prepare,search


def guarded_hit(rows,base_rows):
    assert len(rows)==len(base_rows)
    mean=lambda rs,k: np.mean([r[k] for r in rs])
    deficits=[(mean(base_rows,k)-mean(rows,k))/max(1,abs(mean(base_rows,k))) for k in ('ret','sharpe')]
    base_dd=max(r['dd'] for r in base_rows)
    deficits.append((max(r['dd'] for r in rows)-base_dd)/max(base_dd,1))
    for r,b in zip(rows,base_rows):
        for x,y in [(r['buy'],b['buy']),(r['orders']-r['buy'],b['orders']-b['buy'])]:
            deficits.append((.9*y-x)/max(1,y))
    miss=sum(max(0.,x) for x in deficits if x>1e-10)
    if miss:return -1.-miss
    return sum(r['wins'] for r in rows)/max(1,sum(r['orders'] for r in rows))


def main(a):
    cfg=json.loads((a.prev/'config.json').read_text())
    seed_params=json.loads((a.prev/'locked.json').read_text())['cost']['effective']
    for k,v in seed_params.items():cfg['genes'][k]['baseline']=v
    incumbent=tuple(seed_params[k] for k in cfg['genes'])
    a.out.mkdir(parents=True,exist_ok=True)
    ev,ref,_=prepare(a,cfg);years=cfg['train_years'];vy=cfg['validation_year'];ty=cfg['report_year']
    original={y:ev(incumbent,y,True) for y in years+[vy]}
    score=lambda rows,val:guarded_hit(rows,[original[y] for y in ([vy] if val else years)])
    creator.create('Fit465',base.Fitness,weights=(1.,));creator.create('Strategy464',list,fitness=creator.Fit465)
    runs=[search(cfg,ev,True,seed,a.out,score_fn=score,warm=True) for seed in cfg['seeds']]
    base_hit=score([original[vy]],True)
    best=max(runs,key=lambda r:score([r['val']],True))
    adopt=(score([best['val']],True)>base_hit+cfg['min_delta'] and
           best['train']>score([original[y] for y in years],False)+cfg['min_delta'])
    selected=best['effective'] if adopt else seed_params
    (a.out/'locked.json').write_text(json.dumps(dict(refined=adopt,effective=selected,
        search_runs=runs),indent=2)+'\n')
    report={}
    for name,pars in [('incumbent',seed_params),('selected',selected)]:
        par=tuple(pars[k] for k in cfg['genes']);report[name]={}
        for y in years+[vy,ty]:
            v,book=ref(par,y,True,with_book=True);pnl=np.array([r['pnl'] for r in book])
            pos=pnl[pnl>0];neg=pnl[pnl<0];avgw=pos.mean();avgl=-neg.mean()
            report[name][y]=dict(**ev(par,y,True),wins_buy=v['wins_buy'],avg_win_usd=avgw,
                avg_loss_abs_usd=avgl,breakeven=avgl/(avgw+avgl),profit_factor=pos.sum()/-neg.sum(),
                mean_net_usd=pnl.mean(),zero=int(sum(pnl==0)))
    (a.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('src','bars','prefix','recipe','prev','out'):p.add_argument(k,type=Path)
    main(p.parse_args())
