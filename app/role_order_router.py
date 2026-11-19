"""Market-only head outputs to a single native command; external mode only."""
import math


def role_mode(position):
    if position not in (-1,0,1):raise ValueError('position direction')
    return {0:'flat',1:'long',-1:'short'}[position]


def greedy_command(position, scores, *, entries_allowed=True, force_flat=False):
    """Use only the eligible role. Flat ties abstain, open ties hold; never reverse."""
    mode=role_mode(position)
    if force_flat:return 3 if position else 0
    if mode=='flat' and not entries_allowed:return 0
    row=scores[mode]
    if len(row)!=(2 if mode=='flat' else 1) or not all(math.isfinite(v) for v in row):
        raise ValueError('eligible role logits')
    if mode!='flat':return 3 if row[0]>0 else 0
    buy,sell=row
    if buy>max(sell,0):return 1
    if sell>max(buy,0):return 2
    return 0
