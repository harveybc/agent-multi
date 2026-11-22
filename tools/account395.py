"""Chronological native training-market adapter; evaluation stays on the original market."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from app.frozen_recipe_inputs import FrozenRecipeInputs


def stitch_inputs(left,right):
    if (left.recipe!=right.recipe or not len(left.row_ts_ns) or not len(right.row_ts_ns)
            or left.row_ts_ns[-1]>=right.row_ts_ns[0]):
        raise ValueError("recipe or decision-clock discontinuity")
    arrays=[]
    for key in ("row_ts_ns","slot_ts_ns","windows","valid","period"):
        x=np.concatenate([getattr(left,key),getattr(right,key)])
        x.flags.writeable=False
        arrays.append(x)
    return FrozenRecipeInputs(left.recipe,*arrays)


def take(inputs,mask):
    return FrozenRecipeInputs(inputs.recipe,*(getattr(inputs,k)[mask] for k in
        ("row_ts_ns","slot_ts_ns","windows","valid","period")))


def stitch_market(left,right,calendar):
    if left.recipe!=right.recipe or left.config!=right.config:
        raise ValueError("different business rules or representation")
    bars=pd.concat([left.bars,right.bars])
    if (not bars.index.is_monotonic_increasing or not bars.index.is_unique or
            (calendar.assign(bars.index)<0).any()):
        raise ValueError("overlap or outside declared sessions")
    market=copy.copy(right)
    market.inputs=stitch_inputs(left.inputs,right.inputs)
    market.bars=bars
    market.calendar=calendar
    # Frozen per-origin causal estimates, not refit at concatenation. In
    # particular a synthetic/real change in absolute price is not volatility.
    for k in ("sigma","sigma_rule"):
        x=np.concatenate([getattr(left,k),getattr(right,k)])
        if len(x)!=len(bars):
            raise ValueError("scale/bar alignment")
        setattr(market,k,x)
    return market


def replay(m,risk,interval):
    from app.three_signal_learning import environment
    env=environment(m,risk,6,interval,.99)
    obs=env.reset()
    trace=[]
    while not env.finished:
        # Deterministic witness, not an agent profitability experiment.
        direction=int(np.sign(obs["features"][-1,0]))
        action={-1:2,0:0,1:1}[direction]
        _,reward,_,duration=env.step(np.int64(action))
        trace.append((env.bar,action,reward,duration,env.engine.balance))
        obs=env.last_observation
    return np.asarray(trace),np.asarray(env.equity_trace)


def run(a):
    from app.selected_input_market import market_for
    spec=json.loads((a.parent/"SPEC.json").read_text())
    recipe=json.loads(a.recipe.read_text())
    m=market_for(a.parent,spec,recipe)
    risk=json.loads((a.parent/"RISK.json").read_text())
    # Split inside a session, not on a weekend: an account reset here matters.
    cut=pd.Timestamp("2022-01-12 12:00",tz="UTC")
    halves=[]
    for before in (True,False):
        other=copy.copy(m)
        ix=(m.bars.index<cut)==before
        other.bars=m.bars[ix];other.sigma=m.sigma[ix];other.sigma_rule=m.sigma_rule[ix]
        other.inputs=take(m.inputs,(m.inputs.row_ts_ns<(cut+pd.Timedelta(hours=1)).value)==before)
        halves.append(other)
    rebuilt=stitch_market(*halves,m.calendar)
    interval=["2022-01-02","2022-01-22"]
    a1,e1=replay(m,risk,interval); a2,e2=replay(rebuilt,risk,interval)
    np.testing.assert_array_equal(a1,a2);np.testing.assert_array_equal(e1,e2)
    if np.allclose(e1[:,1],e1[0,1]) or not (e1[0,0]<cut.value<e1[-1,0]):
        raise ValueError("vacuous boundary witness")
    try:
        stitch_market(halves[1],halves[0],m.calendar)
    except ValueError:
        pass
    else:
        raise ValueError("reversed market accepted")
    rep=dict(stage="NATIVE_ONE_ACCOUNT_STITCH_WITNESS",data="real-only, split and rejoined",
             width=recipe["list"]["n"],decisions=len(a1),equity_rows=len(e1),cut=str(cut),
             initial_cash=float(e1[0,1]),final_cash=float(e1[-1,1]),max_equity_delta=0,
             max_reward_delta=0,sac_training=False)
    a.out.mkdir(exist_ok=False,parents=True)
    (a.out/"RESULT.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(rep)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("parent","recipe","out"): p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
