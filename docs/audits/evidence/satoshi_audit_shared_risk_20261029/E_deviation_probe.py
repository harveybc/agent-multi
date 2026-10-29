"""Task E: does deviation change ZigZag pivots in the pinned heuristic-strategy zigzag.py?"""
import hashlib, sys
from pathlib import Path
import numpy as np
ENGINE = Path(sys.argv[1])
sys.path.insert(0, str(ENGINE))
from app.short_horizon import zigzag as Z
src = ENGINE/'app/short_horizon/zigzag.py'
print('zigzag.py sha256', hashlib.sha256(src.read_bytes()).hexdigest())

rng = np.random.default_rng(20261028)
n = 200
close = 1.1 + np.cumsum(rng.normal(0, .0008, n))
high = close + rng.uniform(0, .0012, n)
low = close - rng.uniform(0, .0012, n)

def pivots(depth, backstep, dev):
    return Z.hindsight_pivots(high, low, Z.Params(depth=depth, deviation_pips=dev, backstep=backstep))

for depth, backstep in ((3, 1), (4, 2), (7, 6), (12, 11)):
    ref = pivots(depth, backstep, 0.0)
    print(f'\ndepth={depth} backstep={backstep}: {len(ref)} pivots at deviation=0; first 12 idx={[p[0] for p in ref[:12]]}')
    for dev in (0.0, 5.0, 50.0, 1e6, -1.0):
        pv = pivots(depth, backstep, dev)
        same = pv == ref
        maps = Z.candidate_maps(high, low, Z.Params(depth=depth, deviation_pips=dev, backstep=backstep))
        mref = Z.candidate_maps(high, low, Z.Params(depth=depth, deviation_pips=0.0, backstep=backstep))
        print(f'  deviation={dev:>8}: {len(pv)} pivots, identical_to_dev0={same}, candidate_maps_identical={all(np.array_equal(a, b) for a, b in zip(maps, mref))}')

# Why: the deviation test only runs when the window extreme changed; it rejects when low[i]-val > dev,
# but line 77 only emits a candidate when low[i] == val (difference 0 <= dev for any dev >= 0).
# Residual channel: when rejected, the backstep cleanup loop is skipped. Probe whether cleanup ever clears
# something that would otherwise not be cleared: brute force over random short paths.
hits = 0; trials = 0
for _ in range(3000):
    m = int(rng.integers(6, 30))
    c = np.cumsum(rng.normal(0, 1, m)); h = c + rng.uniform(0, 1.5, m); l = c - rng.uniform(0, 1.5, m)
    d = int(rng.integers(2, 8)); b = int(rng.integers(1, d))
    a = Z.candidate_maps(h, l, Z.Params(depth=d, deviation_pips=0.0, backstep=b))
    z = Z.candidate_maps(h, l, Z.Params(depth=d, deviation_pips=1e9, backstep=b))
    trials += 1
    if not all(np.array_equal(x, y) for x, y in zip(a, z)):
        hits += 1
        if hits <= 3:
            print('  DIFF found on random path: depth', d, 'backstep', b, 'len', m)
print(f'\nrandom brute force: {trials} paths, candidate maps differ for deviation 0 vs 1e9 in {hits} paths')
# Negative deviation (out of contract) as a positive control that the comparison line is reached:
diffneg = 0
for _ in range(500):
    m = int(rng.integers(6, 30))
    c = np.cumsum(rng.normal(0, 1, m)); h = c + rng.uniform(0, 1.5, m); l = c - rng.uniform(0, 1.5, m)
    d = int(rng.integers(2, 8)); b = int(rng.integers(1, d))
    a = Z.candidate_maps(h, l, Z.Params(depth=d, deviation_pips=0.0, backstep=b))
    z = Z.candidate_maps(h, l, Z.Params(depth=d, deviation_pips=-1e9, backstep=b))
    diffneg += not all(np.array_equal(x, y) for x, y in zip(a, z))
print(f'positive control (deviation=-1e9, out of contract): candidate maps differ in {diffneg}/500 paths -> the comparison line is live')
