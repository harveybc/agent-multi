"""Detect target-row reuse through the history of cross-fitted features."""
import numpy as np


def require_scope_report(report, lane_identity, raw_sha256, prepared_sha256):
    if (report.get('lane_identity')!=lane_identity or report.get('raw_sha256')!=raw_sha256
            or report.get('prepared_receipt_sha256')!=prepared_sha256):
        raise ValueError('STL_ACTIVITY_AUDIT_LINEAGE')
    if not report.get('passed') or report['reports']['seasonal']['paired_evaluation_rows_with_overlap']!=0:
        raise ValueError('STL_ACTIVITY_CONTEXT_OVERLAP')


def context_fit_overlap(fit_rows, prediction_rows, available, offsets):
    available=np.asarray(available,dtype=bool)
    n=len(available)
    if available.ndim!=1 or len(fit_rows)!=len(prediction_rows)+1:
        raise ValueError('FEATURE_SCOPE_SHAPE')
    offsets=np.asarray(offsets)
    if offsets.ndim!=1 or not len(offsets) or not np.issubdtype(offsets.dtype,np.integer) or (offsets<0).any():
        raise ValueError('FEATURE_SCOPE_OFFSETS')
    owners=np.zeros(n,dtype=int);assigned=np.zeros(n,dtype=bool)
    fit=np.zeros((len(fit_rows),n),dtype=bool)
    def rows(value):
        value=np.asarray(value)
        if value.ndim!=1 or not np.issubdtype(value.dtype,np.integer) or ((value<0)|(value>=n)).any():
            raise ValueError('FEATURE_SCOPE_ROWS')
        return value
    for i,value in enumerate(fit_rows):fit[i,rows(value)]=True
    for i,value in enumerate(prediction_rows,1):
        value=rows(value)
        if assigned[value].any() or len(np.unique(value))!=len(value):raise ValueError('FEATURE_SCOPE_DUPLICATE')
        assigned[value]=True;owners[value]=i
    overlap=np.zeros(n,dtype=bool)
    first_lag=np.full(n,-1,dtype=int)
    all_rows=np.arange(n)
    for lag in np.unique(offsets):
        target=all_rows[all_rows>=lag];source=target-lag
        touched=target[available[source] & fit[owners[source],target]]
        first_lag[touched[~overlap[touched]]]=lag
        overlap[touched]=True
    return overlap,first_lag,owners
