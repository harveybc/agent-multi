"""Deterministic FIT-only choice among already finite-verified corrections."""


def best_rescue(candidates):
    valid=[r for r in candidates if r['accepted']]
    return min(valid,key=lambda r:(r['regret'],r['cost'],r['row'])) if valid else None
