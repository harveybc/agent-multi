"""FIT-only native continuation targets on fixed, authenticated anchor support."""
import torch

from app.all_state_refinement import validate_panel
from app.decision_attribution import continuation, state_view


def validate_anchor_identity(expected, actual):
    keys=('row','timestamp','window_start','role')
    if (not expected or len({r['timestamp'] for r in expected})!=len(expected)
            or [[r[k] for k in keys] for r in expected]!=[[r[k] for k in keys] for r in actual]):
        raise ValueError('NATIVE_ANCHOR_IDENTITY')


def target_values(critic, records, kind):
    if kind not in ('critic','native_fit'):raise ValueError('NATIVE_TARGET_KIND')
    if critic.shape!=(len(records),3):raise ValueError('NATIVE_TARGET_SHAPE')
    result=critic.detach().clone()
    for i,r in enumerate(records):
        if r['role'] not in ('fit','monitor'):raise ValueError('NATIVE_TARGET_ROLE')
        if kind=='native_fit' and r['role']=='fit':
            result[i]=torch.tensor([o['reward_return'] for o in r['outcomes']],
                                   dtype=result.dtype,device=result.device)
    if not torch.isfinite(result).all():raise ValueError('NATIVE_TARGET_FINITE')
    return result


def rebuild_panel(make, inputs, signals, anchors, progress):
    intervals=dict(fit=['2020-01-01','2023-01-01'],monitor=['2023-01-01','2024-01-01'])
    validate_anchor_identity(anchors,anchors)
    wanted={r['timestamp']:r for r in anchors}; records=[]
    for role,interval in intervals.items():
        env=make(interval);env.reset()
        terminal=int(env.engine.v[env.engine.last])+3600*10**9
        while not env.finished:
            row=int(env.rows[env.bar]);ts=int(inputs.row_ts_ns[row])
            if ts in wanted:
                old=wanted[ts]
                rec={k:old[k] for k in ('year','timestamp','window_start','row','role')}
                if old['role']!=role or old['row']!=row or old['window_start']!=int(inputs.slot_ts_ns[row,0]):
                    raise ValueError('NATIVE_ANCHOR_REPLAY')
                before=state_view(env)
                outcomes=[continuation(env,signals,action) for action in (0,1,2)]
                if state_view(env)!=before:raise ValueError('NATIVE_PARENT_MUTATED')
                rec.update(terminal=terminal,state=before,
                    position=0 if env.engine.pos is None else env.engine.pos.direction,
                    outcomes=outcomes,allowed=all(o['first_effect']['entries']>0 for o in outcomes[1:]))
                records.append(rec);progress(anchors=len(records),total=len(anchors))
            env.step(int(signals[row]))
    validate_anchor_identity(anchors,records);validate_panel(records)
    return records
