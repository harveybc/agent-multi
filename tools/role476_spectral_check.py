"""Replay the archived spectral producers at four historical decision boundaries."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from role476_restart_audit import digest


def run(snapshot, data):
    sys.path.insert(0, str(snapshot.resolve()))
    from tools import layer1_pool_h1 as l1, layer2_pool_h1 as l2

    prices = pd.read_parquet(data / "OHLC.parquet")
    store = pd.read_parquet(data / "FEATURES.parquet")
    names = [c for c in store if c.startswith(("wv.", "ssa.", "hb."))]

    def rebuild(bars):
        p = l2.parents(bars.rename(columns=dict(OPEN="BidOpen", HIGH="BidHigh",
            LOW="BidLow", CLOSE="BidClose")).rename_axis("timestamp").reset_index())
        ssa = l1.causal_ssa(bars.CLOSE.to_numpy())
        x = pd.concat([l2.fam_wavelet(p)[0], l2.fam_hilbert(p)[0],
            pd.DataFrame(ssa[:, 1:3] / l1.PIP, index=bars.index,
                         columns=["ssa.c2_pips", "ssa.c3_pips"])], axis=1)
        result = {}
        for name in names:
            base, _, op = name.partition("@")
            s = x[base]
            result[name] = {"diff": s.diff, "log": lambda: np.log(s), "": lambda: s}[op]()
        return pd.DataFrame(result)

    evidence = []
    for year in range(2016, 2020):
        part = prices.loc[str(year)]
        mid = len(part) // 2
        bars = part.iloc[mid-300:mid+50]
        t = bars.index[300]
        ref = rebuild(bars).loc[t]
        cut = rebuild(bars.loc[:t]).loc[t]
        altered = bars.copy()
        altered.loc[altered.index > t] *= 1.025
        suffix = rebuild(altered).loc[t]
        altered.loc[t] *= 1.0125
        now = rebuild(altered).loc[t]
        truth = store.loc[t, names]
        if not np.isfinite([ref, cut, suffix, now, truth]).all():
            raise ValueError(f"incomplete at {t}")
        families = {}
        for family in ("wv", "ssa", "hb"):
            cols = [n for n in names if n.startswith(family + ".")]
            e = lambda other: float((ref[cols]-other[cols]).abs().max())
            families[family] = dict(channels=len(cols), archived_error=e(truth),
                                    truncated_error=e(cut), suffix_error=e(suffix),
                                    current_bar_effect=e(now))
            if max(e(truth), e(cut), e(suffix)) > 1e-8 or e(now) <= 1e-8:
                raise ValueError((str(t), family, families[family]))
        evidence.append(dict(bar_start=t.isoformat(), families=families))
    return dict(cuts=evidence, producer_sha256={f:digest(snapshot / "tools" / f)
                for f in ("layer1_pool_h1.py", "layer2_pool_h1.py", "lib/native_wavelet.py")})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("snapshot", "data", "output"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    result = run(a.snapshot, a.data)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
