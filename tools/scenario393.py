"""NON_GOVERNING: fit and sample a regime/segment script, never market bars."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def lengths(rows):
    cuts = np.flatnonzero(np.asarray(rows.start)[1:] != np.asarray(rows.end)[:-1]) + 1
    return np.diff(np.r_[0, cuts, len(rows)])


def context(rows, close):
    a = np.asarray(rows.start)
    if min(a) < 48:
        raise ValueError("insufficient macro context")
    # Trading-time windows ending at each segment origin, independent of its future.
    r = np.diff(np.log(close[a[:, None] + np.arange(-48, 1)]), axis=1)
    trend = r.sum(1) / np.maximum(np.abs(r).sum(1), 1e-12)
    return np.column_stack([trend, np.log(np.maximum(r.std(1), 1e-10)),
                            np.log(np.maximum(r[:, -12:].std(1), 1e-10))])


def fit_states(cat, x):
    years = pd.to_datetime(cat.end_ns, utc=True).dt.year
    prefix = years < 2022
    valid = (years == 2022) & (pd.to_datetime(cat.start_ns, utc=True).dt.year == 2022)
    train = years < 2023
    scale = StandardScaler().fit(x[prefix])
    def fit(k, seed, mask):
        return GaussianHMM(k, covariance_type="diag", n_iter=150, random_state=seed,
                           tol=.01).fit(scale.transform(x[mask]), lengths(cat[mask]))
    trials = []
    for k in range(1, 7):
        for seed in (17, 43):
            m = fit(k, seed, prefix)
            ll = m.score(scale.transform(x[valid]), lengths(cat[valid])) / valid.sum()
            if np.isfinite(ll):
                trials.append((float(ll), k, seed))
    _, k, seed = max(trials)
    scale.fit(x[train])
    m = fit(k, seed, train)
    donors = cat[train].copy()
    donors["state"] = m.predict(scale.transform(x[train]), lengths(donors))
    check = (years == 2023) & (pd.to_datetime(cat.start_ns, utc=True).dt.year == 2023)
    one = fit(1, 17, train)
    iid = GaussianMixture(k, covariance_type="diag", n_init=8, random_state=17).fit(
        scale.transform(x[train]))
    score = lambda model: float(model.score(scale.transform(x[check]), lengths(cat[check])) / check.sum())
    return donors, dict(k=k, seed=seed, fits=trials, check_2023=score(m),
                        single_state_2023=score(one),
                        iid_same_k_2023=float(iid.score(scale.transform(x[check]))))


def dynamics(rows):
    start = rows.start.to_numpy()[1:] != rows.end.to_numpy()[:-1]
    chain = np.r_[0, start.cumsum()]
    state = rows.state.to_numpy()
    spells = rows.assign(chain=chain, run=np.r_[0, (start | (state[1:] != state[:-1])).cumsum()])
    runs = spells.groupby(["chain", "run"], sort=False).state.agg(["first", "size"]).reset_index()
    initial = runs.groupby("chain", sort=False).head(1)["first"].to_numpy()
    inside = runs.chain.eq(runs.chain.shift()) & runs.chain.eq(runs.chain.shift(-1))
    # Do not treat the unknown ends of catalog chains as complete sojourns.
    distribution = {s: g["size"].to_numpy() for s, g in runs[inside].groupby("first")}
    edges = {}
    for i in np.flatnonzero(runs.chain.eq(runs.chain.shift(-1))):
        edges.setdefault(runs["first"][i], []).append(runs["first"][i+1])
    return initial, distribution, edges


def compose(rows, clock, seed):
    if not len(clock) or clock.has_duplicates or not clock.is_monotonic_increasing:
        raise ValueError("ambiguous calendar")
    rng = np.random.default_rng(seed)
    initial, duration, edges = dynamics(rows)
    fields = ["displacement", "vol", "retrace", "retrace_at", "curvature", "first", "middle", "last"]
    profiles = {(s, d): g for (s, d), g in rows.groupby(["state", "side"])}
    cutoff = clock[0]
    gaps = np.r_[False, np.diff(clock.as_unit("ns").asi8) != 3600 * 10**9].cumsum()
    if clock.tz is None or cutoff.year != 2016 or clock[-1].year != 2019:
        raise ValueError("unexpected calendar")
    one = rows.state.nunique() == 1
    state = int(rng.choice(initial))
    side = int(rng.choice(rows.side))
    i, episode, result = 0, 0, []
    while i < len(clock):
        n = len(clock) if one else int(rng.choice(duration[state]))
        for _ in range(n):
            g = profiles[state, side]
            p = g.iloc[rng.integers(len(g))]
            count = min(int(p.n), len(clock)-i)
            result.append(dict(episode=episode, regime=state, side=side, start_row=i,
                               start=clock[i], end=clock[i+count-1], bars=count,
                               desired_bars=int(p.n), censored=count!=int(p.n),
                               reopenings=int(gaps[i+count-1]-(gaps[i-1] if i else 0)),
                               profile_origin_ns=int(p.start_ns), **{f:float(p[f]) for f in fields}))
            i += count
            side *= -1
            if i == len(clock):
                break
        if i < len(clock):
            state = int(rng.choice(edges[state]))
        episode += 1
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["catalog", "bank", "clock", "out"]:
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    cat = pd.read_parquet(args.catalog)
    with np.load(args.bank) as f:
        ts, close = f["ts"], f["real"][:, 3]
    if not (np.array_equal(ts[cat.start], cat.start_ns) and
            np.array_equal(ts[cat.end], cat.end_ns) and pd.to_datetime(ts, utc=True).year.max() == 2023):
        raise ValueError("catalog/bars identity")
    clock = pd.DatetimeIndex(pd.read_parquet(args.clock, columns=["Time"]).Time)
    clock = clock[(clock.year >= 2016) & (clock.year < 2020)]
    donors, metrics = fit_states(cat, context(cat, close))
    plan = compose(donors, clock, args.seed)
    pd.testing.assert_frame_equal(plan, compose(donors, clock, args.seed))
    if plan.bars.sum() != len(clock) or (plan.side.diff().iloc[1:] == 0).any():
        raise ValueError("broken sequence")
    if not plan.start.equals(pd.Series(clock[plan.start_row])) or not np.array_equal(
            plan.end, clock[plan.start_row+plan.bars-1]):
        raise ValueError("calendar coverage")
    args.out.mkdir(exist_ok=False, parents=True)
    dest = args.out/"SCRIPT.parquet"
    plan.to_parquet(dest, index=False)
    donors.to_parquet(args.out/"TRAIN_PROFILES.parquet", index=False)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    metrics.update(stage="MACRO_SCRIPT_ONLY", calendar="observed H1; not certified session coverage",
                   seed=args.seed, bars=len(clock), origins=len(cat), donors=len(donors),
                   segments=len(plan), spells=int(plan.episode.nunique()), calendar_sha256=hashlib.sha256(
                       clock.as_unit("ns").asi8.tobytes()).hexdigest(),
                   hashes={p.name:digest(p) for p in (args.catalog, args.bank, dest)})
    (args.out/"REPORT.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(json.dumps({k:v for k,v in metrics.items() if k not in ("fits","hashes")}))


if __name__ == "__main__":
    main()
