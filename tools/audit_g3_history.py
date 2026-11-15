"""Measure candidate joint primitive history without admitting training data."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.g3_producer_audit import identity, require_grid, support_report
from tools.audit_g3_producers import authenticate, load_external, write
from tools.recover_stl_price_warmup import sha


def history_panel(spec):
    data = load_external(spec)
    cross = data['cross']
    end = pd.Timestamp(spec['train_end'])
    native = pd.read_parquet(spec['producer_settings']['eurusd'],
                             filters=[('timestamp','<',end)]).set_index('timestamp')
    require_grid(native)
    params = data['cfg']['params']
    parts = [native[['open','high','low','close']].add_prefix('eurusd.'),
             pd.DataFrame({p+'.close':s.reindex(native.index) for p,s in data['fx'].items()},
                          index=native.index)]
    for name,series in data['daily'].items():
        features = cross.daily_features(series,name,data['cfg']['daily'][name]['kind'])
        chosen = {'us10y':['dly.us10y.chg_1d','dly.us10y.chg_5d'],
                  'dax':['dly.dax.ret_5d'],'spx':['dly.spx.ret_1d']}[name]
        parts.append(cross.asof_join(native.index,features[chosen],cross.daily_available(
            features.index,params['daily_lag_days'],params['release_offset_hours'][name]),
            pd.Timedelta(days=params['daily_max_stale_days'])))
    cot = cross.cot_features(data['cot'])[['cot.eur.net_long','cot.eur.net_share','cot.eur.net_z156']]
    parts.append(cross.asof_join(native.index,cot,cross.cot_available(cot.index,
        params['release_offset_hours']['cot']),pd.Timedelta(days=params['cot_max_stale_days'])))
    return pd.concat(parts,axis=1)


def summarize(panel,train_start):
    require_grid(panel)
    before = panel.loc[panel.index < train_start]
    if before.empty:
        return dict(candidate_rows=0,finite_rows=0,first=None,last=None,columns=len(panel.columns))
    support = support_report(before,list(before),[0])
    # A gap or missing member invalidates a candidate window; no forward-fill.
    return dict(candidate_rows=len(before),finite_rows=support['finite_rows']['count'],
                first=support['finite_rows']['first'],last=support['finite_rows']['last'],
                columns=len(panel.columns),missing_by_column={
                    c:int((~np.isfinite(before[c].to_numpy(float))).sum()) for c in before},
                support_sha256=support['window_support_sha256'])


def run(spec_path,output):
    spec = json.loads(spec_path.read_text())
    if not spec.get('external'):
        raise ValueError('G3_JOINT_SOURCES_REQUIRED')
    authenticate(spec)
    panel = history_panel(spec)
    report = dict(schema='g3_candidate_history.v1',status='CANDIDATE_COVERAGE_ONLY',
                  spec_sha256=sha(spec_path),code_sha256=sha(__file__),
                  helpers={name:sha(Path(__file__).parent/name) for name in ('audit_g3_producers.py','recover_stl_price_warmup.py')},
                  support_helper_sha256=sha(Path(__file__).resolve().parents[1]/'app/g3_producer_audit.py'),
                  candidate_before_train=summarize(panel,pd.Timestamp(spec['train_start'])),
                  columns=list(panel),training_released=False,real_history_extended=False,
                  source_certified=False,calendar_admitted=False,learned_features_rebuilt=False,
                  note='Legacy source identities/publication rules retained. Complete raw/derived71 history, calendar admission and producer fit scopes are not certified by finite primitive rows.')
    report['identity'] = identity(report)
    authenticate(spec)
    if output.exists():
        if json.loads(output.read_text()) != report:
            raise ValueError('G3_HISTORY_OUTPUT_IDENTITY')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        write(output,report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    run(a.spec,a.output)
