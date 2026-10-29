"""Paired exit metrics on a fixed position; future-aware evaluation, not inputs."""
import numpy as np

from app.optimal_exit import targets


def _inputs(close_net, passive_net, risk_usd, tolerance):
    x=np.asarray(close_net,dtype=float)
    if (x.ndim!=1 or not np.isfinite(x).all()
            or not np.isfinite([passive_net,risk_usd,tolerance]).all()
            or risk_usd<=0 or tolerance<0):
        raise ValueError('INVALID_EXIT_METRIC_INPUT')
    return x


def evaluate_exit(close_net, passive_net, risk_usd, chosen_index, *, actual_net=None, tolerance=1e-7):
    x=_inputs(close_net,passive_net,risk_usd,tolerance)
    if chosen_index is not None and (type(chosen_index) is not int or not 0<=chosen_index<len(x)):
        raise ValueError('INVALID_EXIT_INDEX')
    planned=float(passive_net if chosen_index is None else x[chosen_index])
    actual=planned if actual_net is None else float(actual_net)
    if not np.isfinite(actual) or abs(actual-planned)>tolerance:
        raise ValueError('NATIVE_EXIT_MISMATCH')
    best=max(float(passive_net),float(x.max()) if len(x) else float(passive_net))
    improvement=actual-float(passive_net)
    opportunity=best-float(passive_net)
    regret=max(0.,best-actual)  # Only sub-tolerance native rounding can put actual above best.
    early=chosen_index is not None
    return dict(actual_net_usd=actual,passive_net_usd=float(passive_net),optimal_net_usd=best,
        initial_risk_usd=float(risk_usd),improvement_usd=improvement,opportunity_usd=opportunity,
        regret_usd=regret,regret_risk=regret/risk_usd,
        captured_fraction=improvement/opportunity if opportunity>tolerance else None,
        early_close=early,beneficial_close=early and improvement>tolerance,
        destructive_close=early and improvement < -tolerance,
        neutral_close=early and abs(improvement)<=tolerance,
        missed_opportunity=opportunity>tolerance and regret>tolerance,
        missed_without_closing=not early and opportunity>tolerance,
        executable_close_count=len(x),chosen_index=chosen_index,tolerance_usd=tolerance)


def tick_metrics(close_net, passive_net, risk_usd, *, tolerance=1e-7):
    x=_inputs(close_net,passive_net,risk_usd,tolerance)
    if not len(x):
        return []
    t=targets(x,passive_net,risk_usd)
    return [dict(index=i,close_net=float(x[i]),best_remaining_net=float(t['value'][i]),
                 best_continuation_net=float(t['hold_value'][i]),
                 close_regret_remaining_risk=float((t['value'][i]-x[i])/risk_usd),
                 close_advantage_usd=float(t['close_advantage'][i])) for i in range(len(x))]


def summarize_exits(rows, *, tolerance=1e-7):
    n=len(rows)
    improvement=sum(r['improvement_usd'] for r in rows)
    opportunity=sum(r['opportunity_usd'] for r in rows)
    early=sum(r['early_close'] for r in rows)
    bad=sum(r['destructive_close'] for r in rows)
    return dict(positions=n,early_closes=early,beneficial_closes=sum(r['beneficial_close'] for r in rows),
        destructive_closes=bad,neutral_closes=sum(r['neutral_close'] for r in rows),
        destructive_share_of_early=bad/early if early else None,
        missed_opportunities=sum(r['missed_opportunity'] for r in rows),
        missed_without_closing=sum(r['missed_without_closing'] for r in rows),
        improvement_usd_sum=improvement,improvement_usd_mean=improvement/n if n else None,
        opportunity_usd_sum=opportunity,aggregate_capture=improvement/opportunity if opportunity>tolerance else None,
        regret_risk_mean=sum(r['regret_risk'] for r in rows)/n if n else None,
        regret_risk_max=max((r['regret_risk'] for r in rows),default=None))
