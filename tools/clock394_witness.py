"""Independent EURUSD H1 witness for the disputed hours of the 4+4 prefix."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pandas as pd


def run(disputes, out):
    out.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(disputes, parse_dates=["utc"])
    hist, meta = {}, {}
    for stamp in evidence.utc:
        # Vendors number rollover weeks differently. Check adjacent archives
        # and require observations on both sides before reporting nonappearance.
        iso = stamp.isocalendar()
        weeks = {(stamp.year, n) for n in range(max(1, iso.week-2), min(53, iso.week+2)+1)}
        if stamp.month == 12:
            weeks.update((stamp.year, n) for n in (50, 51, 52, 53))
            weeks.update((stamp.year+1, n) for n in (1, 2))
        seen = set()
        for year, week in sorted(weeks):
            tag = f"{year}-{week}"
            if tag not in hist:
                url = f"https://candledata.fxcorporate.com/H1/EURUSD/{year}/{week}.csv.gz"
                cache = out / f"{tag}.csv.gz"
                try:
                    if not cache.exists():
                        with urlopen(Request(url,headers={"User-Agent":"research-calendar-witness/1.0"}), timeout=12) as r:
                            raw=r.read()
                        gzip.decompress(raw)
                        cache.write_bytes(raw)
                    raw=cache.read_bytes()
                    df=pd.read_csv(io.BytesIO(gzip.decompress(raw)))
                    ts=pd.to_datetime(df.DateTime,format="%m/%d/%Y %H:%M:%S.%f",utc=True)
                    hist[tag]=set(ts)
                    meta[tag]=dict(url=url,rows=len(ts),sha256=hashlib.sha256(raw).hexdigest())
                except HTTPError as e:
                    if e.code not in (403,404):
                        raise
                    hist[tag]=set()
                    meta[tag]=dict(url=url,http=e.code)
            seen.update(hist[tag])
        left=max((x for x in seen if x<stamp),default=None)
        right=min((x for x in seen if x>stamp),default=None)
        ix=evidence.utc==stamp
        evidence.loc[ix,"fxcm_witness"]=("quoted" if stamp in seen else
            "not_in_archive" if left and right else "no_coverage")
        evidence.loc[ix,"previous_fxcm"]=str(left)
        evidence.loc[ix,"next_fxcm"]=str(right)
    evidence.to_csv(out/"WITNESS.csv",index=False)
    (out/"SOURCES.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(evidence[["utc","eurusd","fxcm_witness","previous_fxcm","next_fxcm"]].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("disputes",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    run(a.disputes,a.out)
