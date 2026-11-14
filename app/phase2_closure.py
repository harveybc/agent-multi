"""Bounded final-subset certification using the existing frozen selection rule."""
import math

from app.cumulative_readmission import cumulative_pass


def coverage(universe, blocks, selected, evaluate):
    if (len(set(universe)) != len(universe) or
            sorted(sum(blocks, [])) != sorted(universe) or
            len(set(selected)) != len(selected) or not set(selected) <= set(universe) or
            not selected or any(not b for b in blocks)):
        raise ValueError('CLOSURE_UNIVERSE')
    if any(set(b) & set(selected) and not set(b) <= set(selected) for b in blocks):
        raise ValueError('CLOSURE_PARTIAL_BLOCK')
    current = [c for c in universe if c in selected]

    def measure(cols):
        value = evaluate(cols)
        if type(value['eligible']) is not bool or not math.isfinite(value['score']):
            raise ValueError('CLOSURE_SCORE')
        return value

    baseline = measure(current)
    if not baseline['eligible']:
        raise ValueError('CLOSURE_INACTIVE_BASELINE')
    rows = []
    for block in blocks:
        remove = set(block) <= set(current)
        wanted = set(current) - set(block) if remove else set(current) | set(block)
        candidate = [c for c in universe if c in wanted]
        if not candidate:
            rows.append(dict(block=block, phase='DELETE', candidate=[], feasible=False,
                             qualifies=False, reason='EMPTY_MODEL'))
            continue
        value = measure(candidate)
        qualifies = value['eligible'] and (value['score'] > baseline['score'] or
                                           (remove and value['score'] == baseline['score']))
        rows.append(dict(block=block, phase='DELETE' if remove else 'REINSERT',
                         candidate=candidate, feasible=True, score=value['score'],
                         eligible=value['eligible'], qualifies=qualifies))
    return dict(selected=current, baseline_score=baseline['score'], neighbors=rows,
                qualifying_moves=sum(r['qualifies'] for r in rows))


def close_phase2(universe, blocks, initial, evaluate, max_cycles=2, progress=None):
    if type(max_cycles) is not int or not 0 <= max_cycles <= 2:
        raise ValueError('CLOSURE_BUDGET')
    selected = [c for c in universe if c in initial]
    cycles, checks = [], []
    for n in range(max_cycles + 1):
        check = coverage(universe, blocks, selected, evaluate)
        checks.append(check)
        if progress:
            progress(dict(stage='COVERAGE', cycle=n, check=check))
        if check['qualifying_moves'] == 0:
            status = 'LOCAL_FROZEN_NEIGHBORHOOD_STABLE'
            break
        if n == max_cycles:
            status = 'BUDGET_CLOSED_NOT_LOCAL_OPTIMUM'
            break
        cycle = cumulative_pass(universe, blocks, selected, evaluate, progress)
        cycles.append(cycle)
        if cycle['selected'] == selected:
            raise ValueError('CLOSURE_QUALIFIED_MOVE_NOT_APPLIED')
        selected = cycle['selected']
    return dict(status=status, selected=selected, cycles=cycles, coverage=checks,
                max_corrective_cycles=max_cycles, globally_optimal=False,
                frozen_donor_assisted=True)
