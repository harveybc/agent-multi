"""Inventory and research-only historical rebuild of the 61-channel price arm."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.audit_g3_producers import authenticate, producers, rebuild, sha


def run(a):
    recipe = json.loads(a.recipe.read_text())
    cols = recipe["list"]["columns"]
    spec = json.loads(a.spec.read_text())
    settings = spec["producer_settings"]
    spec["pins"][str(a.factor)] = spec["pins"].pop(settings["factor_state"])
    settings["factor_state"] = str(a.factor)
    authenticate(spec)
    state = json.loads(Path(settings["scope"]).read_text())
    loader = producers(spec)
    q = pd.read_parquet(a.quotes).set_index("Time")[["Open","High","Low","Close"]]
    q.columns = ["OPEN","HIGH","LOW","CLOSE"]
    q.index = q.index.as_unit("ns").rename("timestamp")
    assert q.index.is_unique and q.index.is_monotonic_increasing
    official = pd.read_parquet(a.admitted).set_index("timestamp").loc["2020":"2023"]
    official = official[["BidOpen","BidHigh","BidLow","BidClose"]].set_axis(q.columns,axis=1)
    both = official.index.intersection(q.index)
    d = (official.loc[both] - q.loc[both]).abs().max(axis=1)
    clock = pd.DatetimeIndex(pd.read_parquet(a.calendar/"CLOCK.parquet").timestamp)
    missing = clock.difference(q.index)
    times = clock.intersection(q.index)
    # Never treat data after an unresolved in-session gap as uninterrupted history.
    contiguous = times[times < missing.min()] if len(missing) else times
    seed = q.loc[q.index < clock[0]].tail(512)
    frame = pd.concat([seed,q.loc[contiguous]])
    x = rebuild(frame,cols,state,json.loads((a.calendar/"CLOSURES.json").read_text()),loader)
    x = x.loc[contiguous]
    if len(cols)!=61 or list(x)!=cols or not x.index.equals(contiguous):
        raise ValueError("recipe or clock mismatch")
    def lineage(c):
        if c.startswith(("cal.","ev.")) or c=="px.hours_since_session_open":
            return "clock_calendar"
        if c.startswith(("kf.","rg.gmm.")):
            return "real_price_with_frozen_train_estimator"
        return "real_price_derived"
    a.out.mkdir(exist_ok=False,parents=True)
    x.to_parquet(a.out/"RESEARCH_FEATURES.parquet")
    finite = np.isfinite(x)
    inventory = pd.DataFrame({"column":cols,"dependency":[lineage(c) for c in cols],
                              "finite_rows":finite.sum().to_numpy(),"total_rows":len(x)})
    inventory.to_csv(a.out/"COLUMNS.csv",index=False)
    years = q.loc["2016":"2023"].groupby(q.loc["2016":"2023"].index.year).size()
    rep = dict(year_counts=years.to_dict(), source_sha256=sha(a.quotes), recipe_sha256=sha(a.recipe),
               admitted_sha256=sha(a.admitted), overlap_real_train_rows=len(official),
               overlap_common=len(both), overlap_exact_ohlc=int(d.eq(0).sum()),
               unmatched_train_timestamps=len(official)-len(both),
               groups=inventory.dependency.value_counts().to_dict(), reconstructed_rows=len(x),
               fully_finite_rows=int(finite.all(axis=1).sum()), first=str(x.index[0]),last=str(x.index[-1]),
               admitted_clock_missing_from_source=list(map(str,missing)), sac_release=False,
               note="Research, historical reconstruction only. Frozen 2020-23 transforms, not a 2016 vintage. No imputation, no validation data.")
    (a.out/"RESULT.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("recipe","spec","factor","quotes","admitted","calendar","out"):
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
