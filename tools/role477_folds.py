"""Time-ordered seasonal panels. Rebuild inputs before releasing a training panel."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tools.audit_g3_producers import producers, rebuild, write
from tools.recover_stl_price_warmup import sha

H = pd.Timedelta(hours=1)
def read(p):
    return json.loads(p.read_text())


def prepare(a):
    y = a.year
    start, monitor, end, stop = [pd.Timestamp(s, tz="UTC") for s in
        (f"{y-4}-01-01", f"{y-1}-07-01", f"{y}-01-01", f"{y+1}-01-01")]
    d = a.snapshot
    loaded = producers(dict(core_producer=str(d/'core_selection_bundle.py'), producer_settings=dict(
        compatibility_path=str(d/'stl-producer-compat-20261111'), technical_package=str(d/'pandas_ta'),
        technical_generator=str(d/'normalize_phase1b.py'), predictor_root=str(d/'predictor'))))
    l2 = loaded[2]['layer2_pool_h1']
    names = read(a.recipe)['list']['columns']
    prefix = pd.read_parquet(a.history/'OHLC.parquet')
    warm = pd.read_parquet(a.history/'WARMUP_OHLC.parquet')
    src = pd.read_parquet(a.market/'bars.parquet', filters=[('timestamp', '<', stop-H)])
    if not (src.available_at == src.timestamp+H).all():
        raise ValueError('bar availability')
    rename = dict(OPEN='BidOpen', HIGH='BidHigh', LOW='BidLow', CLOSE='BidClose')
    bars = pd.concat([warm, prefix, src.set_index('timestamp')[list(rename.values())]
                      .rename(columns={v:k for k,v in rename.items()})])
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError('overlapping or unordered sources')
    closures = read(a.history/'CLOSURES.json')
    # The historical calendar covers both sources (including conservative blackouts).
    if pd.Timestamp(closures[-1][1]) < stop-H:
        raise ValueError('calendar coverage')
    p = l2.parents(bars.rename_axis('timestamp').rename(columns=rename).reset_index())
    fit = (p.index+H >= start) & (p.index+H < monitor)
    fp = p.loc[fit]
    st = dict(params={n:fn(fp, np.ones(len(fp), bool)) for n,fn in
                      [('kalman', l2.fit_kalman), ('gmm', l2.fit_gmm)]})
    if not all(v['converged'] for v in st['params'].values()):
        raise ValueError(('parameter fit did not converge', st))
    f = rebuild(bars, names, st, closures, loaded)
    archive = pd.concat([pd.read_parquet(a.history/'FEATURES.parquet', columns=names),
                         pd.read_parquet(a.market/'store.parquet', columns=['timestamp', *names],
                             filters=[('timestamp','<',stop-H)]).set_index('timestamp')])
    # Learned channels change by design. Compare deterministic channels well away
    # from each source's distinct warm-up origin, on rows actually used by the model.
    fixed = [c for c in names if not c.startswith(('rg.gmm.', 'kf.'))]
    with np.load(a.data/'dataset_sdelta.npz') as z:
        stamps = pd.to_datetime(z['ts'], utc=True)
        active = (stamps >= start) & (stamps < stop)
        stamps = stamps[active]
        targets = {k:z[k][active] for k in ('y_short','y_long','ok_short','ok_long')}
    origins = bars.index.get_indexer(stamps-H)
    if np.any(origins < 24) or not stamps.is_unique or not stamps.is_monotonic_increasing:
        raise ValueError('origin coverage')
    slots = origins[:,None] + np.arange(-24,1,2)
    ix = bars.index[np.unique(slots)]
    seam = src.timestamp.iloc[0]
    ix = ix[(ix<seam)|(ix>=src.timestamp.iloc[min(1024,len(src)-1)])]
    actual, old = f.loc[ix,fixed], archive.loc[ix,fixed]
    parity = {}
    for c in fixed:
        v,b = actual[c].to_numpy(),old[c].to_numpy()
        fail = ~np.isclose(v,b,rtol=1e-7,atol=1e-8,equal_nan=True)
        if fail.any():
            parity[c] = dict(rows=int(fail.sum()),first=ix[fail][0].isoformat(),
                             max_abs=float(np.nanmax(abs(v-b))))
    # Boundary checks reproduce the actual stored window (all 13 slots), not only t.
    checks = []
    for cut in (monitor, end):
        row = origins[np.searchsorted(stamps.asi8,cut.value)]
        r = rebuild(bars.iloc[:row+1], names, st, closures, loaded).tail(25).iloc[::2]
        ref = f.loc[r.index]
        np.testing.assert_allclose(r,ref,rtol=1e-9,atol=1e-10)
        checks.append(dict(last_bar=str(r.index[-1]),max_abs=float(np.max(abs(r-ref)))))
    finite = np.isfinite(f).all(axis=1)
    zfit = f.loc[fit & finite]
    mu, sd = zfit.mean(), zfit.std(ddof=0)
    if not (sd>0).all() or len(names)!=61 or len(fixed)!=56:
        raise ValueError('scaler schema')
    x = ((f-mu)/sd).to_numpy(dtype=np.float32)[slots]
    if not np.isfinite(x).all():
        raise ValueError('nonfinite decision window')
    report = dict(year=y,start=str(start), monitor=str(monitor),eval_start=str(end),
        last_fit_close=str(zfit.index[-1]+H),last_eval_close=str(stamps[-1]),fit_bars=len(zfit),
        windows=len(x),prefix_checks=checks,parity_checked_rows=len(ix),parity_failures=parity,
        excluded_from_archive_comparison='first 1024 bars after 2020 source boundary',
        producer_state=st,scaler=dict(columns=names,mean=mu.tolist(),std=sd.tolist()),
        fixed_feature_list_retrospective=True,old_encoder_allowed=False)
    a.out.mkdir(parents=True,exist_ok=True)
    write(a.out/'audit.json',report)
    if parity:
        raise ValueError(('pre-existing and reconstructed fixed indicators disagree',parity))
    tmp = a.out/'batch.partial.npz'
    np.savez(tmp,ts=stamps.asi8,x=x,**targets)
    tmp.replace(a.out/'batch.npz')
    write(a.out/'ready.json',dict(audit=sha(a.out/'audit.json'),batch=sha(a.out/'batch.npz')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('year',type=int,choices=range(2020,2024))
    for k in ('snapshot','recipe','history','market','data','out'):
        p.add_argument(k,type=Path)
    prepare(p.parse_args())
