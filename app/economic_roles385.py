import numpy as np

def best_open(buy,sell,tol=1e-8):
    if not np.isfinite([buy,sell]).all():raise ValueError('nonfinite advantage')
    if max(buy,sell)<=tol:return 0
    if abs(buy-sell)<=tol:return -1
    return 1 if buy>sell else 2
