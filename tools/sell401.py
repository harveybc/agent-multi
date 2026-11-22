"""Join min-twin Q ranks at the actual decision bar to realized SELL fills."""
import json
from pathlib import Path

import numpy as np
import torch
from tools import pilot391 as p
from tools.account397 import assemble
from app.frozen_signal_masks import prepared_signals
from app.three_signal_learning import evaluate


def main():
    root=Path('/var/tmp/sac399-series')
    snap=Path('/var/tmp/sell401')
    parent=Path('/var/tmp/sac70-warm-v3-20261108')
    bundle=root/'model/61/bundle'
    old=Path('/var/tmp/diag400/8y-s7')
    p.configure(7)
    m,_=p.load_bundle(bundle,'cpu')
    _,market=assemble(parent,bundle,root/'history397-61-r2')
    risk=p.read(parent/'RISK.json')
    history={r['epoch']:r['metrics']['VALIDATION'] for r in p.read(snap/'HISTORY.json')}
    result=[]
    for epoch,path in [(0,bundle),(3,old/'bundles/epoch_0003'),(4,snap/'recovery/epoch_0004'),
                       (17,old/'recovery/epoch_0017'),(20,snap/'bundles/epoch_0020'),(21,snap/'bundles/epoch_0021')]:
        ck=path/'model.pt'
        assert p.file_sha(ck)==p.read(path/'MANIFEST.json')['files']['model.pt']
        m.load(ck)
        env=p.environment(market,risk,6,m.recipe['split']['periods']['VALIDATION'],m.effective['gamma'])
        env.reset()
        ix=env.rows[env.bar:env.engine.last][env.valid[env.bar:env.engine.last]]
        logits,q=p.emit(m,market.inputs,ix,twins=True)
        vals=q.min(axis=0)
        metrics=evaluate(env,logits,prepared=prepared_signals(logits))
        ref=history[epoch] if epoch else p.read(root/'model/61/READY.json')['initial']['VALIDATION']
        if ref['trades']!=metrics['trades'] or abs(ref['return_pct']-metrics['return_pct'])>1e-7:
            raise ValueError(('failed endpoint check',epoch,metrics['trades'],ref['trades']))
        sells=[t for t in env.engine.trades if t['direction']==-1]
        rows=np.asarray([env.rows[t['entry_bar_abs']-1] for t in sells],dtype=int)
        assert np.isin(rows,ix).all() and (logits[rows].argmax(1)==2).all()
        ranked=vals[rows].argmax(axis=1)==2
        net=np.array([t['net_usd'] for t in sells])
        result.append(dict(epoch=epoch,sha256=p.file_sha(ck),decisions=len(ix),trades=metrics['trades'],
            total_net=metrics['net_profit_usd'],sells=len(sells),sell_net=net.sum(),
            sell_Q_top_all=int((vals[ix].argmax(1)==2).sum()),sell_Q_top_at_sell=int(ranked.sum()),
            sell_net_when_Q_top=float(net[ranked].sum()),sell_net_other=float(net[~ranked].sum())))
    p.write(snap/'SELL.json',dict(observational=True,checkpoints=result,
        history=[dict(epoch=i,sells=r['sell']['trades'],sell_net=r['sell']['net_profit_usd'],
            trades=r['trades']) for i,r in history.items()]))


if __name__=='__main__': main()
