"""Frozen-panel sampling and same-state continuation diagnostics."""
import numpy as np

QUOTAS = dict(flat_both_allowed=12, open_long=4, open_short=4, flat_restricted=2)


def choose_panel(trace, excluded_timestamps, seed=312):
    if (len({r['timestamp'] for r in trace}) != len(trace)
            or any(r['year'] not in range(2020,2024) or r['kind'] not in QUOTAS for r in trace)):
        raise ValueError('COVERAGE_TRACE')
    rng = np.random.default_rng(seed); chosen = []; counts = {}
    for year in range(2020,2024):
        for kind, quota in QUOTAS.items():
            eligible = [r for r in trace if r['year']==year and r['kind']==kind
                        and r['timestamp'] not in excluded_timestamps]
            indices = rng.permutation(len(eligible))[:quota]
            chosen.extend(eligible[int(i)] for i in indices)
            counts[f'{year}:{kind}'] = dict(available=len(eligible), requested=quota, selected=len(indices))
    return sorted(chosen, key=lambda r:r['timestamp']), counts


def drift_metrics(source, changed, actions):
    x,y = np.asarray(source,float), np.asarray(changed,float)
    a = np.asarray(actions)
    if (x.shape != y.shape or x.ndim!=2 or x.shape[1]!=3 or not len(x)
            or a.shape!=(len(x),) or a.dtype.kind not in 'iu' or not ((a>=0)&(a<3)).all()
            or not np.isfinite([x,y]).all()):
        raise ValueError('COVERAGE_DRIFT_SHAPE')
    xset = x.max(1)[:,None]-x <= 1e-8
    yset = y.max(1)[:,None]-y <= 1e-8
    pairs = ((0,1),(0,2),(1,2))
    delta = np.stack([(y[:,i]-y[:,j])-(x[:,i]-x[:,j]) for i,j in pairs], axis=1)
    ix = np.arange(len(x))
    return dict(n=len(x), return_rmse=float(np.sqrt(np.mean((x-y)**2))),
        gap_rmse=float(np.sqrt(np.mean(delta**2))),
        optimal_set_changed=int(np.any(xset!=yset, axis=1).sum()),
        mean_actor_regret=float(np.mean(y.max(1)-y[ix,a])),
        mean_actor_regret_under_source=float(np.mean(x.max(1)-x[ix,a])),
        actor_optimal=int(yset[ix,a].sum()))


def nearest_fit_rms(windows, fit):
    a,b = np.asarray(windows,float),np.asarray(fit,float)
    if a.shape[1:]!=b.shape[1:] or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('COVERAGE_DISTANCE_INPUT')
    a,b = a.reshape(len(a),-1),b.reshape(len(b),-1)
    distance = np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)
    return np.sqrt(distance.min(1)/a.shape[1])
