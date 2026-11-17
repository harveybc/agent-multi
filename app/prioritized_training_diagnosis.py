"""Read-only numerical summaries. No selection, training or promotion."""
import math
import numpy as np


def summarize_history(history, progress):
    if not history:
        raise ValueError('EMPTY_HISTORY')
    epochs = [r['epoch'] for r in history]
    if epochs != sorted(set(epochs)):
        raise ValueError('NONMONOTONE_HISTORY')
    if epochs[-1] > progress['completed_epochs']:
        raise ValueError('SNAPSHOT_RACE')
    rows = []
    for h in history:
        train, val = (h['metrics'][p] for p in ('TRAIN', 'VALIDATION'))
        numbers = [train['return_pct'], val['return_pct'], h['epoch_seconds']]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('NONFINITE_HISTORY')
        signals = val['signals']
        count = sum(signals.values())
        rows.append(dict(epoch=h['epoch'], train_return_pct=train['return_pct'],
            validation_return_pct=val['return_pct'], gap_pp=train['return_pct']-val['return_pct'],
            validation_trades=val['trades'], validation_orders_per_week=val['orders_per_week'],
            validation_signal_shares={k:v/count if count else None for k,v in signals.items()},
            lr=h.get('learning_rates_used'), epoch_seconds=h['epoch_seconds'],
            evaluation_seconds=h.get('evaluation_seconds')))
    seconds=float(np.median([r['epoch_seconds'] for r in rows[-5:]]))
    decisions=sum(h['decisions'] for h in history)
    updates=history[-1].get('updates')
    # Exclude any partial current epoch from both numerator and denominator.
    ratio=None
    if epochs == list(range(1, progress['completed_epochs']+1)):
        ratio=updates/decisions if updates is not None and decisions else None
    return dict(rows=rows, latest=rows[-1], completed_epochs=progress['completed_epochs'],
        current_epoch=progress.get('epoch'), bad=progress.get('bad'),
        patience_limit=progress.get('patience_limit'), selected_epoch=progress.get('selected_epoch'),
        cumulative_updates=updates, completed_decisions=decisions, updates_per_completed_decision=ratio,
        eta_to_minimum_seconds=max(0,progress['minimum_epochs']-progress['completed_epochs'])*seconds,
        seconds_per_epoch_recent_median=seconds,
        scope='Descriptive existing accounts; signal shares are not replay coverage; no promotion')


def data_profile(x, stamps, valid, periods, columns):
    x=np.asarray(x); stamps=np.asarray(stamps); valid=np.asarray(valid,dtype=bool)
    if x.ndim != 2 or x.shape != (len(stamps),len(columns)):
        raise ValueError('PROFILE_SHAPE')
    if np.any(stamps >= np.datetime64('2025-01-01','ns').astype('int64')):
        raise ValueError('TEST_DATA_FORBIDDEN')
    years=stamps.astype('datetime64[ns]').astype('datetime64[Y]').astype(int)+1970
    fit=valid & (periods=='TRAIN'); val=valid & (periods=='VALIDATION')
    if not fit.any() or not val.any() or not np.isfinite(x[fit|val]).all():
        raise ValueError('PROFILE_SUPPORT')
    out=[]
    for i,name in enumerate(columns):
        a,b=x[fit,i],x[val,i]; sd=float(a.std()); lo,hi=np.quantile(a,[.01,.99])
        per_year={}
        for year in (2020,2021,2022,2023):
            mask=fit & (years==year)
            pairs=mask[1:] & mask[:-1] & (np.diff(stamps)==3600_000_000_000)
            left,right=x[:-1,i][pairs],x[1:,i][pairs]
            corr=(float(np.corrcoef(left,right)[0,1]) if len(left)>1
                  and left.std()>0 and right.std()>0 else None)
            per_year[str(year)]=dict(valid_rows=int(mask.sum()), adjacent_pairs=int(pairs.sum()),lag1=corr)
        out.append(dict(column=name,train_std=sd,constant_train=sd==0,
            validation_mean_shift_train_std=float((b.mean()-a.mean())/sd) if sd else None,
            validation_outside_train_p01_p99=float(((b<lo)|(b>hi)).mean()),years=per_year))
    return dict(columns=out,train_rows=int(fit.sum()),validation_rows=int(val.sum()),
        note='Lag-one dependence is descriptive, not effective sample size or causal attribution')
