"""Retain all retrospective K=3 edges; no account or causal-policy claims."""
import numpy as np

from app.zigzag_opportunities import HOUR, PIP


def expand_edges(tape, pivots, records):
    origins = np.asarray(records['origin'])
    if not np.array_equal(origins, np.flatnonzero(tape.allowed)):
        raise ValueError('ORIGIN_SUPPORT')
    indices = np.array([p[0] for p in pivots] + [len(tape.open)], dtype=int)
    prices = np.array([p[2] for p in pivots] + [np.nan])
    if np.any(np.diff(indices) <= 0):
        raise ValueError('EVENT_ORDER')
    chunk_end = np.empty(len(tape.open), dtype=int)
    for a, b in tape.chunks:
        chunk_end[a:b] = b
    slots = np.minimum(np.searchsorted(indices, origins, side='right')[:,None] + np.arange(3), len(indices)-1)
    ends = indices[slots]
    valid = ends < chunk_end[origins,None]
    complete = valid.all(axis=1)
    ends = np.where(valid, ends, -1)
    if not np.array_equal(ends, records['candidate_end']) or not np.array_equal(complete, records['pair_valid']):
        raise ValueError('SOURCE_EVENT_MISMATCH')
    shape = (2, len(origins), 3)
    net, mae, draw, score = [np.full(shape, np.nan) for _ in range(4)]
    durations = np.full((len(origins),3), np.nan)
    target = np.where(valid, prices[slots], np.nan)
    for r in np.flatnonzero(complete):
        o = int(origins[r])
        for k, end in enumerate(ends[r]):
            duration = (tape.times[end] - tape.times[o]) / HOUR + 1
            durations[r,k] = duration
            cost = tape.fill_cost[o] + tape.fill_cost[end]
            for side, sign in enumerate((1,-1)):
                favorable = tape.high[o:end+1] if sign == 1 else -tape.low[o:end+1]
                adverse = tape.low[o:end+1] if sign == 1 else -tape.high[o:end+1]
                entry = sign*tape.open[o]
                peak = np.maximum.accumulate(np.maximum(entry, favorable))
                dd = float(np.max(peak-adverse))
                a = max(0., float(entry-np.min(adverse)))
                profit = sign*(target[r,k]-tape.open[o])-cost-(tape.financing[side,end]-tape.financing[side,o])
                net[side,r,k], mae[side,r,k], draw[side,r,k] = profit, a, dd
                score[side,r,k] = profit/(max(dd,cost,PIP)*duration)
    error = 0.
    for side in (0,1):
        rows = np.flatnonzero(complete)
        best = np.argmax(score[side,complete],axis=1)
        if not np.array_equal(ends[rows,best],records['end'][side,rows]):
            raise ValueError('SOURCE_WINNER_CHANGED')
        for actual, name in ((net,'net'),(mae,'mae'),(draw,'drawdown'),(score,'score')):
            error = max(error, float(np.max(np.abs(actual[side,rows,best]-records[name][side,rows]))))
    if error > 1e-12:
        raise ValueError(('SOURCE_VALUE_CHANGED', error))
    out = dict(origin=origins, candidate_end=ends, complete=complete, target=target,
               duration=durations, net=net, mae=mae, drawdown=draw, score=score,
               calendar_eligible=complete[:,None] & (ends+1 <= tape.limit[origins,None]))
    summary = dict(origins=len(origins), complete_origins=int(complete.sum()),
        censored_origins=int((~complete).sum()), retained_directional_edges=int(complete.sum()*6),
        nonpositive_edges=int(np.sum(net[:,complete] <= 0)),
        nonwinning_edges=int(complete.sum()*4), original_winner_max_error=error,
        calendar_eligible_directional_edges=int(out['calendar_eligible'].sum()*2))
    return out, summary
