"""Authenticated closed-policy diagnostics, no fits, expert labels or TEST."""
import argparse
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import file_sha, require
from tools.recipe_hybrid_queue import Market, authenticated, verify
from tools.recipe_sac_queue import read, write, now, clean, tensor_hash


def trade_summary(trades):
    durations=[]
    for t in trades:
        require(np.isclose(t['gross_usd']-t['entry_cost_usd']-t['exit_cost_usd'],
                           t['net_usd'],rtol=0,atol=1e-8),'TRADE_ACCOUNTING')
        duration=(pd.Timestamp(t['exit_time'])-pd.Timestamp(t['entry_time'])).total_seconds()/3600
        require(np.isfinite(duration) and duration>=0,'TRADE_DURATION')
        durations.append(duration)
    n=len(trades)
    return dict(n=n,gross_usd=sum(t['gross_usd'] for t in trades),
        cost_usd=sum(t['entry_cost_usd']+t['exit_cost_usd'] for t in trades),
        net_usd=sum(t['net_usd'] for t in trades),
        win_rate=sum(t['net_usd']>0 for t in trades)/n if n else None,
        one_hour_share=sum(x<=1 for x in durations)/n if n else None,
        duration_hours_median=float(np.median(durations)) if n else None,
        directions={'long':sum(t['direction']==1 for t in trades),'short':sum(t['direction']==-1 for t in trades)})


def hold_counterfactual(engine,trade):
    """Same historical position, branch ONLY its discretionary exit decision."""
    from app.short_horizon.expert_guided.env import Position, CLOSE, HOLD
    require(trade['close_reason']=='AGENT_CLOSE','VOLUNTARY_CLOSE_REQUIRED')
    entry,k=int(trade['entry_bar_abs']),int(trade['exit_bar_abs'])
    require(engine.first<=entry<k<=engine.last,'COUNTERFACTUAL_PERIOD')
    require(pd.Timestamp(trade['entry_time']).value==int(engine.v[entry])
            and pd.Timestamp(trade['exit_time']).value==int(engine.v[k]),'COUNTERFACTUAL_CLOCK')
    require(trade['entry_price']==float(engine.o[entry]),'COUNTERFACTUAL_ENTRY')

    def branch():
        e=copy.copy(engine)
        e.reset()
        e.cfg=replace(e.cfg,percentual_order_size=trade['percentual_order_size'])
        e.pos=Position(trade['direction'],trade['units'],entry,trade['entry_price'],
                       trade['stop_loss'],trade['take_profit'],float(e.sigma[entry]),
                       int(e.seg_of[entry]),e._fill_cost(trade['units'],entry,'MARKET'))
        return e

    close=branch();close.step(k-1,CLOSE)
    replay=close.trades[-1]
    for key in ('entry_time','exit_time'):
        require(pd.Timestamp(replay[key])==pd.Timestamp(trade[key]),'CLOSE_REPLAY_CLOCK')
    for key in ('gross_usd','net_usd','entry_cost_usd','exit_cost_usd','units','exit_price',
                'spread_usd','slippage_usd','commission_usd','financing_usd'):
        require(np.isclose(replay[key],trade[key],rtol=0,atol=1e-8),'CLOSE_REPLAY_ACCOUNTING')
    held=branch()
    for b in range(k-1,held.last):
        held.step(b,HOLD)
        if held.pos is None:break
    require(held.pos is None and len(held.trades)==1,'HOLD_TERMINAL_REQUIRED')
    alternative=held.trades[0]
    return dict(entry_bar=entry,decision_bar=k-1,direction=trade['direction'],units=trade['units'],
        close_replay_exact=True,actual_net_usd=trade['net_usd'],hold_net_usd=alternative['net_usd'],
        close_minus_hold_net_usd=trade['net_usd']-alternative['net_usd'],
        hold_reason=alternative['close_reason'],hold_exit_time=str(alternative['exit_time']),
        extra_hours=(alternative['exit_time']-pd.Timestamp(trade['exit_time'])).total_seconds()/3600)


def common_support(market,period):
    env=market.env(period);obs,_=env.reset()
    features=[];bars=[]
    while True:
        if obs['action_mask'][1] and obs['action_mask'][2]:
            features.append(obs['features'].copy());bars.append(env.bar)
        obs,_,done,_,_=env.step([0,.5,.5,.5])
        if done:break
    require(len(features)>1,'COMMON_SUPPORT_REQUIRED')
    ix=np.unique(np.linspace(0,len(features)-1,min(256,len(features))).astype(int))
    x=np.stack([features[i] for i in ix]);b=np.asarray([bars[i] for i in ix],np.int64)
    return x,dict(n=len(ix),available=len(features),bars=b.tolist(),
        support_sha256=hashlib.sha256(b.tobytes()+x.tobytes()).hexdigest())


def policy_probe(model,x):
    import torch
    rows=[]
    model.policy.set_training_mode(False)
    for name,mask in [('flat',[1,1,1,0,0]),('long',[1,0,0,1,0]),('short',[1,0,0,0,1])]:
        obs={'features':x,'action_mask':np.tile(mask,(len(x),1)).astype(np.float32)}
        tensor,_=model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            features=model.policy.features_extractor(tensor)
            pi=model.policy.mlp_extractor.forward_actor(features)
            d=model.policy.get_distribution(tensor)
            p=d.cat.probs.cpu().numpy()
            reversed_obs={**tensor,'features':tensor['features'].flip(1)}
            reversed_p=model.policy.get_distribution(reversed_obs).cat.probs.cpu().numpy()
            commands=d.get_actions(True)[:,0].cpu().numpy().astype(int)
        rows.append(dict(route=name,counts=np.bincount(commands,minlength=5).tolist(),
            probability_mean=p.mean(0).tolist(),probability_std=p.std(0).tolist(),
            mean_entropy=float(-(p*np.log(np.maximum(p,1e-30))).sum(1).mean()),
            temporal_reversal_mean_probability_change=float(np.abs(p-reversed_p).mean()),
            temporal_reversal_changed_argmax=float((commands!=reversed_p.argmax(1)).mean()),
            encoded_std_mean=float(features.std(0).mean()),
            policy_head_saturated_fraction=float((pi.abs()>.99).float().mean())))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    require(not out.exists(),'NEW_AUDIT_REQUIRED')
    require(not out.is_relative_to(root),'AUDIT_OUTSIDE_CAMPAIGN')
    spec=verify(root);sha=file_sha(root/'SPEC.private.json')
    status=read(root/'STATUS.json')
    require(status['state']=='COMPLETE' and len(status['completed'])==len(spec['units']),'CLOSED_CAMPAIGN')
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    market=Market(spec,'CAT_FIXED')
    support={p:common_support(market,p) for p in ('TRAIN','INNER_A')}
    write(out/'IDENTITY.json',dict(spec_sha256=sha,tool_sha256=file_sha(Path(__file__)),
        scope='RETROSPECTIVE_DIAGNOSTIC_NOT_POLICY_OR_PORTFOLIO_PERFORMANCE',
        fits=0,test_read=False,observed_2024_used=False,
        support={p:v[1] for p,v in support.items()},created_at=now()))
    summaries=[];started=time.monotonic()
    engines={p:market.env(p).engine for p in support}
    for unit in spec['units']:
        path=root/status['completed'][unit];result=authenticated(path,sha)
        folder=path.parent;report=dict(unit=unit,result_sha256=file_sha(path),periods={},probes=[])
        for period,(x,meta) in support.items():
            trades=read(folder/(period+'.trades.json'))
            aggregate=trade_summary(trades)
            native=next(r for r in result['reports'] if r['period']==period)
            require(np.isclose(aggregate['net_usd'],native['net_profit_usd'],rtol=0,atol=1e-7),'NATIVE_NET_RECONCILIATION')
            reasons=sorted({t['close_reason'] for t in trades})
            counter=[hold_counterfactual(engines[period],t) for t in trades if t['close_reason']=='AGENT_CLOSE']
            write(out/unit/(period+'.counterfactuals.json'),clean(counter))
            n=len(counter)
            report['periods'][period]=dict(all=aggregate,
                by_exit={k:trade_summary([t for t in trades if t['close_reason']==k]) for k in reasons},
                close_counterfactual=dict(n=n,beneficial=sum(t['close_minus_hold_net_usd']>1e-8 for t in counter),
                    harmful=sum(t['close_minus_hold_net_usd'] < -1e-8 for t in counter),
                    mean_close_minus_hold_net_usd=float(np.mean([t['close_minus_hold_net_usd'] for t in counter])) if n else None))
        for step in (25600,51200,76800,102400):
            archive=folder/f'model-{step}.zip'
            require(archive.name in result['artifacts'],'CHECKPOINT_NOT_AUTHENTICATED')
            model=PPO.load(archive,device='cpu');before=tensor_hash(model)
            report['probes'].append(dict(step=step,archive_sha256=file_sha(archive),
                training=read(folder/f'TRAINING_{step}.json')['metrics'],
                periods={p:policy_probe(model,x) for p,(x,_) in support.items()}))
            require(tensor_hash(model)==before,'PROBE_CHANGED_WEIGHTS')
            del model
        write(out/unit/'AUDIT.json',clean(report));summaries.append(report)
        write(out/'STATUS.json',dict(state='RUNNING',units=len(summaries),total=len(spec['units']),
                                    unit=unit,heartbeat_at=now(),seconds=time.monotonic()-started))
        print(unit,'complete',round(time.monotonic()-started,1),flush=True)
    write(out/'SUMMARY.json',clean(summaries))
    write(out/'CLOSE.json',dict(state='COMPLETE',spec_sha256=sha,finished_at=now(),
        units=len(summaries),seconds=time.monotonic()-started,
        artifacts={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*.json')}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
