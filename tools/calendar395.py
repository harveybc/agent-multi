"""Construct the evidence-backed part of the synthetic clock independently of quote gaps."""
import argparse
import json
from pathlib import Path

import pandas as pd


START = pd.Timestamp("2016-01-01", tz="UTC")
# Last complete session before the first unresolved dated holiday schedule.
LIMIT = pd.Timestamp("2018-12-21 22:00", tz="UTC")
SOURCE16 = "https://img.fxcorporate.com/global/pdf/2016_holiday_hours_sch_de.pdf"
SOURCE17 = "https://www.fxcm.com/il/client-notifications/עדכון-לסוחר-24-12/"


def schedule():
    ticks = pd.date_range("2015-10-01", "2018-12-24", inclusive="left", freq="h", tz="UTC")
    ny = ticks.tz_convert("America/New_York")
    tradable = ((ny.weekday < 4) | ((ny.weekday == 4) & (ny.hour < 17)) |
                ((ny.weekday == 6) & (ny.hour >= 17)))
    # The initial UTC New Year blackout follows the native business contract.
    for l, r in [
        ("2016-01-01", "2016-01-02"),
        ("2016-12-25 22:00", "2016-12-27 00:00"),
        ("2017-01-01 22:00", "2017-01-03 00:00"),
        ("2017-12-24 22:00", "2017-12-26 06:00"),
        ("2017-12-31 22:00", "2018-01-02 06:00"),
    ]:
        tradable &= ~((ticks >= pd.Timestamp(l, tz="UTC")) & (ticks < pd.Timestamp(r, tz="UTC")))
    closed = pd.Series(~tradable, index=ticks)
    starts = closed.index[closed & ~closed.shift(fill_value=False)]
    ends = closed.index[~closed & closed.shift(fill_value=False)]
    if len(starts) != len(ends):
        raise ValueError("unbounded closure")
    return ticks[tradable & (ticks >= START) & (ticks < LIMIT)], list(zip(starts, ends, strict=True))


def materialize(quotes, plan, out):
    clock, intervals = schedule()
    observed = pd.DatetimeIndex(pd.read_parquet(quotes, columns=["Time"],
        filters=[("Time", ">=", START), ("Time", "<", LIMIT)]).Time)
    old = pd.read_parquet(plan)
    # Keep the scenario draws fixed; assign them to the new eligible clock.
    rows = old[old.start_row < len(clock)].copy()
    rows["bars"] = [min(n,len(clock)-i) for i,n in zip(rows.start_row,rows.desired_bars)]
    rows["start"] = clock[rows.start_row]
    rows["end"] = clock[rows.start_row+rows.bars-1]
    rows["censored"] = rows.bars != rows.desired_bars
    gaps=pd.Series(clock).diff().gt(pd.Timedelta(hours=1)).cumsum().to_numpy()
    rows["reopenings"]=[int(gaps[i+n-1]-(gaps[i-1] if i else 0))
                        for i,n in zip(rows.start_row,rows.bars)]
    if rows.bars.sum()!=len(clock):
        raise ValueError("script coverage")
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame({"timestamp":clock}).to_parquet(out/"CLOCK.parquet",index=False)
    rows.to_parquet(out/"SCRIPT.parquet",index=False)
    (out/"CLOSURES.json").write_text(json.dumps([[a.isoformat(),b.isoformat()] for a,b in intervals])+"\n")
    rep=dict(status="GENERATION_ONLY",from_utc=str(START),stop_exclusive=str(LIMIT),
             rows=len(clock), source16=SOURCE16, source17=SOURCE17,
             configured_hours_absent_from_quotes=list(map(str,clock.difference(observed))),
             quoted_hours_excluded_by_business_calendar=list(map(str,observed.difference(clock))))
    (out/"REPORT.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(len(clock),len(rep["configured_hours_absent_from_quotes"]),len(rep["quoted_hours_excluded_by_business_calendar"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("quotes","plan","out"): p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    materialize(a.quotes,a.plan,a.out)
