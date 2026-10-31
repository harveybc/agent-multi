"""Native economic evaluation, causal synthetic worlds and stopping contract."""
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from app.dynamic_account_panel import World
from app.train_native_pareto import bid_environment_class
from env_plugins.discrete_curve_native import HOUR
from env_plugins.three_signal_native import ThreeSignalEnv


class Patience:
    def __init__(self, minimum, patience, delta):
        self.minimum, self.patience, self.delta = minimum, patience, delta
        self.best, self.best_any, self.bad, self.inactive = None, None, 0, 0

    def observe(self, value, eligible, step):
        if not np.isfinite(value):
            raise ValueError('STOPPING_NONFINITE')
        improved_any = self.best_any is None or value > self.best_any + self.delta
        if improved_any: self.best_any = value
        improved = eligible and (self.best is None or value > self.best + self.delta)
        if improved:
            self.best, self.bad = value, 0
        elif step >= self.minimum and self.best is not None:
            self.bad += 1
        if step >= self.minimum and self.best is None:
            self.inactive += 1
        else:
            self.inactive = 0
        return dict(improved=bool(improved), improved_any=bool(improved_any),
                    stop=step >= self.minimum and max(self.bad,self.inactive) >= self.patience,
                    bad=self.bad, inactive=self.inactive,
                    stop_reason='no_eligible_checkpoint' if self.inactive>=self.patience else 'patience')


def environment(market, risk, horizon, interval, gamma):
    start,end = [pd.Timestamp(t,tz='UTC')-pd.Timedelta(hours=1) for t in interval]
    engine = bid_environment_class()(market.bars,market.calendar,market.sigma,market.sigma_rule,
                                    start,end,market.config,tick=.00001)
    return ThreeSignalEnv(engine,market.inputs,risk,horizon,gamma)


def synthetic_env(recipe, risk, seed, mirror, gamma, flat=False):
    old = World(dict(seed=seed,pattern='trend',volatility='decreasing',mirror=1),risk).env
    rng = np.random.default_rng(seed)
    n = len(old.v)
    signal, i, phase = np.zeros(n), 0, 0
    while i < n:
        length = int(rng.integers(8,19))
        signal[i:i+length] = (1,0,-1,0)[phase % 4] * mirror if not flat else 0
        phase += 1; i += length
    # The current observed exogenous regime drives the next price increment.
    close = 1.1 + np.r_[0., np.cumsum(signal[:-1] * .00012)]
    op = np.r_[close[0],close[:-1]]
    bars = pd.DataFrame(dict(OPEN=op,HIGH=np.maximum(op,close)+.000005,
                            LOW=np.minimum(op,close)-.000005,CLOSE=close),index=old.bars.index)
    engine = bid_environment_class()(bars,old.cal,np.full(n,.001),np.full(n,.001),
        bars.index[0],bars.index[-1]+pd.Timedelta(hours=1),old.cfg,tick=.00001)
    count = recipe['tensor']['window']['observations']
    width = recipe['list']['n']
    raw = np.zeros((n,width),np.float32)
    raw[:,0] = signal
    if width > 1: raw[1:,1] = np.diff(close)/.00012
    windows = np.zeros((n,count,width),np.float32)
    valid = np.arange(n) >= (count-1)*2
    for b in np.flatnonzero(valid): windows[b] = raw[b-np.arange(count-1,-1,-1)*2]
    inputs = SimpleNamespace(row_ts_ns=engine.v+HOUR,valid=valid,windows=windows,
                              observation=lambda row: {'features':windows[row].copy()})
    return ThreeSignalEnv(engine,inputs,risk,6,gamma), signal


def emit(model, inputs, indices=None, twins=False):
    indices = np.flatnonzero(inputs.valid) if indices is None else indices
    logits = np.full((len(inputs.valid),3),np.nan,np.float32)
    q = np.full((2,len(inputs.valid),3),np.nan,np.float32) if twins else None
    with torch.inference_mode():
        for a in range(0,len(indices),256):
            ix = indices[a:a+256]
            x = torch.tensor(inputs.windows[ix],device=model.device)
            logits[ix] = model.surfaces(x).cpu().numpy()
            if twins: q[:,ix] = model.q_surfaces(x).cpu().numpy()
    return logits,q


def evaluate(env, logits, save=None, diagnose=False):
    obs = env.reset()
    actions = []
    signals, commands = Counter(), Counter()
    entropy, command_entropy = [], []
    closes = []
    while not env.finished:
        row = int(env.rows[env.bar]); raw = logits[row]
        if not np.isfinite(raw).all(): raise ValueError('INVALID_SIGNAL_LOGITS')
        ties = np.flatnonzero(np.isclose(raw,raw.max(),atol=1e-10,rtol=0))
        action = int(ties[0]) if len(ties)==1 else 0
        p = np.exp(raw-raw.max()); p = p/p.sum()
        direction = 0 if env.engine.pos is None else env.engine.pos.direction
        if diagnose and direction and action == (2 if direction==1 else 1):
            from app.suppressed_close_counterfactuals import fixed_position_exits
            position = env.engine.pos
            closes.append(fixed_position_exits(env.engine,abs(position.entry_price-position.sl)*position.units))
        pc = p if not direction else np.array([p[0]+p[1 if direction==1 else 2],p[2 if direction==1 else 1]])
        entropy.append(float(-np.sum(p*np.log(np.maximum(p,1e-30)))))
        command_entropy.append(float(-np.sum(pc*np.log(np.maximum(pc,1e-30)))))
        stamp = int(env.engine.v[env.bar])+HOUR
        obs,reward,done,duration = env.step(action)
        signals[str(action)] += 1; commands[str(env.last_command)] += 1
        actions.append(dict(timestamp=stamp,position=direction,signal=action,command=env.last_command,
                            reward=reward,duration=duration))
    e = env.engine; trades = e.trades
    trace = np.asarray(env.equity_trace)
    peak = np.maximum.accumulate(trace[:,1])
    daily = pd.Series(trace[:,1],index=pd.to_datetime(trace[:,0].astype(np.int64),utc=True)).resample('1D').last().ffill()
    returns = daily.pct_change().dropna().to_numpy()
    profit = e.balance-e.cfg.capital
    cash_error = abs(profit-sum(t['net_usd'] for t in trades))
    if cash_error > 1e-6: raise ValueError('SIGNAL_CASH_ERROR')
    weeks = len(set(e.seg_of[e.first:e.last+1]))
    side = lambda d: dict(trades=sum(t['direction']==d for t in trades),
        wins=sum(t['direction']==d and t['net_usd']>0 for t in trades),
        net_profit_usd=sum(t['net_usd'] for t in trades if t['direction']==d))
    metrics = dict(net_profit_usd=float(profit),return_pct=float(100*profit/e.cfg.capital),
        max_drawdown_pct=float(100*np.max((peak-trace[:,1])/peak)),
        sharpe=float(np.sqrt(365)*returns.mean()/returns.std(ddof=1)) if len(returns)>1 and returns.std(ddof=1)>0 else None,
        trades=len(trades),win_rate=sum(t['net_usd']>0 for t in trades)/len(trades) if trades else None,
        buy=side(1),sell=side(-1),weeks=weeks,orders_per_week=len(trades)/weeks,
        return_per_week=float(100*profit/e.cfg.capital/weeks),
        close_reasons=dict(Counter(t['close_reason'] for t in trades)),
        early_by_side={str(d):sum(t['direction']==d and t['close_reason']=='AGENT_CLOSE' for t in trades) for d in (-1,1)},
        median_holding_bars=float(np.median([t['exit_bar_abs']-t['entry_bar_abs'] for t in trades])) if trades else None,
        costs={k:float(sum(t[k] for t in trades)) for k in ('spread_usd','commission_usd','slippage_usd','financing_usd')},
        signals=dict(signals),commands=dict(commands),cash_error=float(cash_error),
        mean_signal_entropy=float(np.mean(entropy)),mean_requested_command_entropy=float(np.mean(command_entropy)),
        native_counts=e.counts)
    if diagnose:
        metrics['early_close_diagnostic'] = dict(proposals=len(closes),
            harmful=sum(r['delta_vs_hold'] < -1e-8 for r in closes),
            helpful=sum(r['delta_vs_hold'] > 1e-8 for r in closes),
            preempted=sum(r['preempted'] for r in closes),
            mean_regret_risk=float(np.mean([r['regret_risk'] for r in closes])) if closes else None,
            scope='fixed position versus passive protection; not account continuation')
    if save is not None:
        from tools.run_discrete_curve_probe import write
        import json
        save.mkdir(parents=True,exist_ok=True)
        write(save/'METRICS.json',metrics)
        write(save/'TRADES.json',json.loads(json.dumps(trades,default=str)))
        write(save/'ACTIONS.json',actions)
        if diagnose: write(save/'CLOSE_COUNTERFACTUALS.json',closes)
        np.savez_compressed(save/'EQUITY.npz',timestamp=trace[:,0].astype(np.int64),equity=trace[:,1])
    return metrics
