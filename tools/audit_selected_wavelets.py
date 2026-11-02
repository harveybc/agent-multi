"""Replay the pinned trailing-wavelet producer against selected consumer columns."""
import argparse
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import file_sha
from tools.run_discrete_curve_probe import read, write


def run(args):
    spec = read(args.parent/'SPEC.private.json'); paths = spec['inputs']
    for key, path in paths.items():
        if file_sha(path) != spec['input_sha256'][key]: raise ValueError('INPUT_PIN')
    manifest = read(args.layer2_manifest)
    if file_sha(args.cross_asset) != manifest['cross_asset_input']['sha256']:
        raise ValueError('CROSS_ASSET_PIN')
    source = subprocess.check_output(['git','-C',str(args.producer_repo),'show',
        '46af988e:tools/lib/native_wavelet.py'])
    module = {}; exec(compile(source, 'pinned_native_wavelet.py', 'exec'), module)
    compute = module['compute_wavelet_native']
    end = pd.Timestamp('2024-01-01', tz='UTC')
    bars = pd.read_parquet(paths['bars'], filters=[('timestamp','<',end)])
    ca = pd.read_parquet(args.cross_asset, columns=['timestamp','fx.usd_strength.ret_1'],
                         filters=[('timestamp','<',end)]).set_index('timestamp')
    recipe = read(paths['recipe'])
    cols = [c for c in recipe['list']['columns'] if c.startswith('wv.')]
    store = pd.read_parquet(paths['store'], columns=['timestamp']+cols,
                            filters=[('timestamp','<',end)]).set_index('timestamp')
    lc = np.log(bars.BidClose.to_numpy())
    parents = dict(lc=lc, r1=np.r_[np.nan,np.diff(lc)],
                   usd=ca.reindex(bars.timestamp)['fx.usd_strength.ret_1'].to_numpy())
    checks = []; parity = {}
    cuts = [1000, 6000, int(np.searchsorted(pd.DatetimeIndex(bars.timestamp).asi8,
             pd.Timestamp('2023-07-01',tz='UTC').value))]
    def calculate(x):
        return compute(pd.DataFrame({'x':x}),window=256,wavelet='db4',level=4,
            frozen_from={'fold_id':'audited','train_end':'fixed producer settings'},
            sample_interval_seconds=3600,column='x',mode='symmetric')[0].iloc[:,1:]
    rng = np.random.default_rng(2741)
    for tag,x in parents.items():
        base = calculate(x)
        for c in [c for c in cols if c.startswith(f'wv.{tag}.')]:
            suffix = c.split('.')[-1].split('@')
            v = base['wavelet_native_'+suffix[0]].to_numpy().copy()
            if len(suffix)>1:
                if suffix[1]=='diff': v=np.r_[np.nan,np.diff(v)]
                elif suffix[1]=='log': v=np.log(v)
                else: raise ValueError('UNREVIEWED_TRANSFORM')
            rebuilt = pd.Series(v,index=bars.timestamp).reindex(store.index).to_numpy()
            actual = store[c].to_numpy()
            exact = np.array_equal(rebuilt,actual,equal_nan=True)
            parity[c] = dict(exact=bool(exact),max_abs=float(np.nanmax(np.abs(rebuilt-actual))))
            if not exact: raise ValueError('WAVELET_STORE_PARITY:'+c+':'+str(parity[c]))
        for cut in cuts:
            prefix = calculate(x[:cut+1]).to_numpy()
            y = x.copy(); y[cut+1:] = rng.normal(size=len(y)-cut-1)
            y = np.r_[y,rng.normal(size=50)]
            changed = calculate(y).iloc[:cut+1].to_numpy()
            ctrl = x[:cut+1].copy(); ctrl[-1] += .1
            positive = calculate(ctrl).iloc[-1].to_numpy()
            ok = np.array_equal(prefix,changed,equal_nan=True)
            ok = ok and np.array_equal(prefix,base.iloc[:cut+1].to_numpy(),equal_nan=True)
            sensitive = not np.array_equal(prefix[-1],positive,equal_nan=True)
            if not (ok and sensitive): raise ValueError('WAVELET_CAUSAL_TEST')
            checks.append(dict(parent=tag,cut=cut,prefix_exact=True,positive_control=True))
    result = dict(passed=True,producer_blob_sha256=hashlib.sha256(source).hexdigest(),
        recipe_sha256=file_sha(paths['recipe']),store_sha256=file_sha(paths['store']),
        selected_columns=len(cols),parity=parity,checks=checks,
        scope='selected wavelet producer and pointwise/difference transforms; upstream selection exposure unchanged',
        no_2024_or_test_rows=True)
    write(args.output,result)
    print({'passed':True,'columns':len(cols),'checks':len(checks)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','producer-repo','layer2-manifest','cross-asset','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
