"""E: common-action vs sample-specific foresight; nonlinear mean-path counterexamples; two positions."""
import copy, json, sys
from pathlib import Path
AM, HS = sys.argv[1], sys.argv[2]
sys.path.insert(0, AM); sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from app.close_hold_controller import PositionView, PathReport, CloseController, CloseConfig, value_paths

z3 = np.zeros(3); z2 = np.zeros(2)


def rep(paths, w=None):
    a = np.asarray(paths, float)  # [S,H,4]
    S, H = a.shape[:2]
    return PathReport(10, 10, 10, 1, 10 + H, a, np.zeros((S, H)), np.full(S, 1 / S) if w is None else np.asarray(w), 'causal')


def pos(d=1, sl=99., tp=106., ident='p'):
    return PositionView(ident, d, 100., 1., sl, tp, 1., 0., 0., 0., 0., 0.)


def bar(o, h, l, c): return [o, h, l, c]


def decide(p, r, z=0.):
    H = r.paths.shape[1]
    return CloseController(CloseConfig(uncertainty_penalty=z)).decide(p, r, r.issued_at, np.zeros(H), np.zeros(H))


def mean_report(r):
    return rep(np.tensordot(r.weights, r.paths, axes=1)[None])


out = {}
# Long entry 100, SL 99, TP 110, zero costs, risk 1. bar0 open = close-now fill.
# E1: BOTH samples touch SL (at different bars); the weighted-mean path never touches SL.
A = [bar(100, 100.5, 98.5, 100.5), bar(100.5, 102.5, 100.5, 102.5), bar(102.5, 102.5, 102.5, 102.5)]
B = [bar(100, 100, 99.6, 99.6), bar(99.6, 99.6, 98, 98), bar(98, 98, 98, 98)]
# E2: only ONE sample touches SL; the mean path never touches SL.
A2 = [bar(100, 100, 98.5, 98.5), bar(98.5, 98.5, 97, 97), bar(97, 97, 97, 97)]
B2 = [bar(100, 101.4, 99.6, 101.4), bar(101.4, 101.6, 101.4, 101.6), bar(101.6, 101.6, 101.6, 101.6)]
for name, (a, b) in dict(E1_both_touch_mean_never=(A, B), E2_one_touches_mean_never=(A2, B2)).items():
    r = rep([a, b]); m = mean_report(r); P = pos(1, 99., 110.)
    v = value_paths(P, r, z3, z3); vm = value_paths(P, m, z3, z3)
    out[name] = dict(mean_path_min_low=float(np.min(m.paths[0, :, 2])), sl=99.,
        per_sample_values=v.tolist(), expected_values=(r.weights @ v).tolist(), mean_path_values=vm[0].tolist(),
        controller=decide(P, r)['action'], controller_adv=decide(P, r)['advantage_risk'],
        mean_path_controller=decide(P, m)['action'], mean_path_adv=decide(P, m)['advantage_risk'])
# E3: sample-specific foresight (E over samples of max_j W_sj) vs one common action (max_j E W_j).
A = [bar(100, 100, 100, 100), bar(101, 101, 101, 101), bar(98, 98, 98, 98)]
B = [bar(100, 100, 100, 100), bar(98, 98, 98, 98), bar(101, 101, 101, 101)]
r = rep([A, B]); P = pos(1, 97., 110.); v = value_paths(P, r, z3, z3)
fore = float(np.max(v[:, 1:], axis=1).mean()); common = float(np.max(r.weights @ v[:, 1:]))
out['E3_foresight_vs_common'] = dict(values=v.tolist(), E_of_max_wait_foresight=fore, max_of_E_wait_common=common,
    close_now=float(r.weights @ v[:, 0]), controller=decide(P, r)['action'], controller_adv=decide(P, r)['advantage_risk'],
    foresight_rule_would='HOLD' if fore > float(r.weights @ v[:, 0]) else 'CLOSE')
# E4: code-level: delta columns are common actions -> check advantage equals min_j over column means.
rng = np.random.default_rng(29102026); worst = 0.
for _ in range(500):
    S, H = rng.integers(1, 6), rng.integers(1, 8)
    c = 100 + np.cumsum(rng.normal(0, 1, (S, H)), axis=1)
    o = np.concatenate([np.full((S, 1), 100.), c[:, :-1]], axis=1)
    hi = np.maximum(o, c) + rng.uniform(0, 1, (S, H)); lo = np.minimum(o, c) - rng.uniform(0, 1, (S, H))
    w = rng.dirichlet(np.ones(S)); z = float(rng.uniform(0, 2))
    r = rep(np.stack([o, hi, lo, c], 2), w)
    p = pos(int(rng.choice([-1, 1])), 0, 0)
    p = PositionView('x', p.direction, 100., 1., 100 - p.direction * 2.5, 100 + p.direction * 4., 2., 0., 0., 0., 0., 0.)
    v = value_paths(p, r, np.zeros(H), np.zeros(H))
    D = v[:, [0]] - v[:, 1:]
    mu = w @ D; sd = np.sqrt(w @ (D - mu) ** 2)
    ref = float(np.min(mu - z * sd)) / p.risk
    got = CloseController(CloseConfig(uncertainty_penalty=z)).decide(p, r, 10, np.zeros(H), np.zeros(H))['advantage_risk']
    worst = max(worst, abs(ref - got))
out['E4_random_formula_identity'] = dict(trials=500, max_abs_err=worst)
# E5: two positions, same report: tight vs wide stop (long and short), plus a nonzero-cost variant.
for d in (1, -1):
    path = [bar(100, 100, 100, 100), bar(100, 100, 100, 100), bar(100, 100.3, 99.7, 100)]
    mk = lambda lv: 100 + d * (lv - 100)
    P = [[(lambda o,h,l,c: bar(o,h,l,c) if d==1 else bar(o,l,h,c))(*[mk(x) for x in b]) for b in [[100.5, 100.6, 98.7, 98.8], [98.8, 104.5, 98.8, 104.5]]]]
    r = rep(P)
    tight = PositionView('t', d, 100., 1., mk(99.), mk(106.), 1., 0., 0., 0., 0., 0.)
    wide = PositionView('w', d, 100., 1., mk(98.), mk(104.), 2., 0., 0., 0., 0., 0.)
    out[f'E5_two_positions_d{d}'] = dict(tight=decide(tight, r)['action'], wide=decide(wide, r)['action'],
        tight_values=value_paths(tight, r, z2, z2).tolist(), wide_values=value_paths(wide, r, z2, z2).tolist())
print(json.dumps(out, indent=1))
