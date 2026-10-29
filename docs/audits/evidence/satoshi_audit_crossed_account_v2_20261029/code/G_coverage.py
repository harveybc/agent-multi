"""G/H: generator coverage, bank disjointness, price-path hashes across cost/quality, report error
sharing, foresight/causality witnesses, and dispersion calibration per rung."""
import hashlib, json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from native_lib import *
from app.crossed_account_panel import panel_specs, PATTERNS, ENVELOPES, QUALITIES as QS
H = 3600*10**9
out = {}
# --- 1. my re-implementation of the generator, extracting parameters
def my_market(spec):
    p = PATTERNS.index(spec['pattern']); v = ENVELOPES.index(spec['volatility'])
    rng = np.random.default_rng(np.random.SeedSequence([spec['seed'], p, 7401]))
    t = np.arange(120.); strength = rng.uniform(.000025, .0001); par = dict(strength=strength)
    if p == 0:
        amp = rng.uniform(.0002, .0007); per = rng.uniform(3, 9)
        sig = np.cumsum(np.full(120, strength))+amp*np.sin(t/per); par.update(amp=amp, period=per)
    elif p == 1:
        cuts = sorted(rng.choice(np.arange(25, 96), 2, replace=False)); mult = rng.uniform(.6, 1.5)
        dr = np.full(120, strength); dr[cuts[0]:cuts[1]] = -strength*mult; sig = np.cumsum(dr)
        par.update(cuts=[int(x) for x in cuts], reversal_mult=mult)
    else:
        sig = 0; amps = []
        for period in (12, 36, 96):
            a = rng.uniform(.0003, .0015); ph = rng.uniform(0, 2*np.pi); amps.append(a); sig = sig+a*np.sin(2*np.pi*t/period+ph)
        par.update(amps=amps)
    inn = rng.normal(0, .00018, 120); width = rng.uniform(.00003, .00016, 120)
    env = (np.ones(120), np.linspace(.3, 1.8, 120), np.linspace(1.8, .3, 120))[v]
    close = 1.1+sig+np.cumsum(inn*env); opens = np.r_[close[0], close[:-1]]; gaps = []
    for k in sorted(rng.choice(np.arange(15, 104), 3, replace=False)):
        j = rng.choice((-1, 1))*rng.uniform(.0003, .0018); close[k:] += j; opens[k:] += j; gaps.append((int(k), float(j)))
    hi = np.maximum(opens, close)+width*env; lo = np.minimum(opens, close)-width*env
    if spec['mirror'] == -1: opens, hi, lo, close = 2.2-opens, 2.2-lo, 2.2-hi, 2.2-close
    par['gaps'] = gaps; par['innov'] = inn
    return np.column_stack((opens, hi, lo, close)), par
cov = {}
for bank in ('calibration', 'confirmation'):
    sp = panel_specs(bank, 'correlated'); eq = 0; rows = []
    for s in sp:
        X, par = my_market(s); eq += np.array_equal(X, market(s))
        if s['volatility'] == 'constant' and s['spread_pips'] == .5 and s['mirror'] == 1: rows.append((s, par))
    st = np.array([p['strength'] for _, p in rows]); gp = np.array([abs(g[1]) for _, p in rows for g in p['gaps']])
    sg = np.array([np.sign(g[1]) for _, p in rows for g in p['gaps']])
    inn = np.array([p['innov'] for _, p in rows]); C = np.corrcoef(inn); off = C[~np.eye(len(C), dtype=bool)]
    cov[bank] = dict(cells=len(sp), my_generator_equal=int(eq), draws=len(rows),
        seeds=sorted({s['seed'] for s in sp}), trend_strength_pips_per_bar=dict(min=float(st.min()*1e4), max=float(st.max()*1e4), by_pattern={pt: [round(p['strength']*1e4, 3) for s, p in rows if s['pattern'] == pt] for pt in PATTERNS}),
        reversal_cuts=[p['cuts'] for s, p in rows if s['pattern'] == 'reversal'], reversal_mult=[round(p['reversal_mult'], 3) for s, p in rows if s['pattern'] == 'reversal'],
        overlap_amps_pips=[[round(a*1e4, 2) for a in p['amps']] for s, p in rows if s['pattern'] == 'overlap'],
        gaps_per_world=3, gap_size_pips=dict(min=float(gp.min()*1e4), max=float(gp.max()*1e4), median=float(np.median(gp)*1e4)),
        gap_sign_up_share=float((sg > 0).mean()), innovation_corr_across_draws=dict(max_abs=float(np.abs(off).max()), mean=float(off.mean())),
        volatility_envelopes=list(ENVELOPES), spreads_pips=sorted({s['spread_pips'] for s in sp}), mirrors=sorted({s['mirror'] for s in sp}))
out['coverage'] = cov
# smoke banks
sm = json.loads((AM/'docs/audits/evidence/crossed_account_queue_20261029/SMOKE_MANIFEST.json').read_text())
def seeds_of(obj):
    acc = set()
    def walk(x):
        if isinstance(x, dict):
            if 'seed' in x and 'pattern' in x: acc.add((x.get('bank'), x['seed']))
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(obj); return acc
smoke = seeds_of(sm); full = seeds_of(json.loads((RUN/'MANIFEST.json').read_text()))
out['banks'] = dict(smoke_manifest_specs=sorted(map(list, smoke)), full_manifest_bank_seeds=sorted({(b, s) for b, s in full}),
    smoke_full_seed_overlap=sorted({s for _, s in smoke} & {s for _, s in full}),
    calib_conf_overlap=sorted(set(cov['calibration']['seeds']) & set(cov['confirmation']['seeds'])))
# distinct price paths across banks (hash of OHLC)
hs = {}
for bank in ('calibration', 'confirmation'):
    for s in panel_specs(bank, 'correlated'): hs.setdefault(hashlib.sha256(market(s).tobytes()).hexdigest(), []).append((bank, s['seed']))
smk = [dict(s) for b in ('calibration', 'confirmation') for s in execution_specs(b, 'correlated', True)]
for s in smk: hs.setdefault(hashlib.sha256(market(s).tobytes()).hexdigest(), []).append((s['bank'], s['seed']))
out['banks']['distinct_price_hashes'] = len(hs); out['banks']['hash_collisions_across_banks'] = sum(1 for v in hs.values() if len({b for b, _ in v}) > 1)
# --- 2. price path identity across cost/quality on the ENV actually used
grp = {}; nworld = 0
for q in QS:
    for s in panel_specs('confirmation', q):
        w = CrossedWorld(s, CONFIG); e = w.env; nworld += 1
        hh = hashlib.sha256(np.column_stack((e.o, e.h, e.l, e.c)).tobytes()+e.v.tobytes()).hexdigest()
        grp.setdefault((s['seed'], s['pattern'], s['volatility'], s['mirror']), set()).add(hh)
out['price_identity'] = dict(worlds_hashed=nworld, groups=len(grp), groups_with_one_hash=sum(len(v) == 1 for v in grp.values()),
    note='group = (seed,pattern,volatility,mirror) across 5 qualities x 2 spreads = 10 worlds each')
# --- 3. report error sharing (correlated, delayed, omitted; b in a few bars)
def rep(spec, b):
    w = CrossedWorld(spec, CONFIG); r = w.report(b); e = w.env
    fut = np.column_stack((e.o, e.h, e.l, e.c))[b+1:w.end]
    return r, fut, w
share = {}
base = panel_specs('confirmation', 'correlated')
for q in ('correlated', 'delayed', 'omitted_reversal'):
    res = dict(spread_pair_identical=0, mirror_pair_equivariant=0, envelope_pairs_identical=0, seed_pairs_identical=0, checked=0)
    for s in base[:48:12]:
        s = dict(s, quality=q)
        for b in (14, 40, 80):
            r0, f0, _ = rep(s, b); r1, _, _ = rep(dict(s, spread_pips=3.), b); rm, _, _ = rep(dict(s, mirror=-1), b)
            rv, _, _ = rep(dict(s, volatility='increasing'), b); rs, _, _ = rep(dict(s, seed=s['seed']+1), b)
            res['checked'] += 1
            res['spread_pair_identical'] += int(np.array_equal(r0.paths, r1.paths))
            res['mirror_pair_equivariant'] += int(np.allclose(rm.paths, 2.2-r0.paths[:, :, [0, 2, 1, 3]], atol=1e-12, rtol=0))
            res['envelope_pairs_identical'] += int(np.array_equal(r0.paths-f0[None], rv.paths-rep(dict(s, volatility='increasing'), b)[1][None]))
            res['seed_pairs_identical'] += int(np.array_equal(r0.paths, rs.paths))
    share[q] = res
# missing == correlated where available
mm = 0; mc = 0
for s in base[:24:6]:
    wc = CrossedWorld(s, CONFIG); wm = CrossedWorld(dict(s, quality='missing'), CONFIG)
    for b in range(12, 100):
        if wm.report_available(b): mc += 1; mm += int(np.array_equal(wc.report(b).paths, wm.report(b).paths))
share['missing_equals_correlated_when_available'] = [mm, mc]
# mask shared across spread/mirror?
s = base[0]; masks = []
for sp_, mi in itertools.product((.5, 3.), (1, -1)):
    w = CrossedWorld(dict(s, quality='missing', spread_pips=sp_, mirror=mi), CONFIG); masks.append(w.available.copy())
share['missing_mask_identical_across_spread_mirror'] = all(np.array_equal(masks[0], m) for m in masks)
out['report_sharing'] = share
# --- 4. causality / foresight witness: perturb bars after b, recompute report at b
caus = {}
for q in ('ideal', 'correlated', 'delayed', 'omitted_reversal'):
    ch = 0; n = 0
    for s in base[:36:6]:
        for b in (20, 50, 80):
            w = CrossedWorld(dict(s, quality=q), CONFIG); r0 = w.report(b).paths.copy()
            w2 = CrossedWorld(dict(s, quality=q), CONFIG); e = w2.env
            for nm in ('o', 'h', 'l', 'c'):
                x = getattr(e, nm).copy(); x[b+2:] += .003; setattr(e, nm, x)   # shift strictly-future bars (b+2..)
            r2 = w2.report(b).paths
            n += 1; ch += int(not np.allclose(r0, r2))
    caus[q] = dict(reports=n, changed_when_future_bars_shift=ch)
out['causality'] = caus
# --- 5. foresight + dispersion calibration (confirmation, all 360 cells, decision bars 12..100 step 4)
cal = {}
for q in ('ideal', 'correlated', 'delayed', 'omitted_reversal'):
    acc = {j: dict(sgn=[], cov=[], z=[], pred=[], real=[]) for j in (1, 3, 6, 12)}
    for s in panel_specs('confirmation', q):
        w = CrossedWorld(s, CONFIG); c = w.env.c
        for b in range(12, 96, 4):
            r = w.report(b); P = r.paths[:, :, 3]
            for j in acc:
                if j > P.shape[1]: continue
                pred = P[:, j-1]-c[b]; real = c[b+j]-c[b]; m = pred.mean(); sd = pred.std(ddof=0)
                a = acc[j]; a['pred'].append(m); a['real'].append(real); a['sgn'].append(np.sign(m) == np.sign(real))
                a['cov'].append(P[:, j-1].min() <= c[b+j] <= P[:, j-1].max()); a['z'].append(abs(real-m)/max(sd, 1e-12))
    cal[q] = {str(j): dict(n=len(a['pred']), corr_pred_real=float(np.corrcoef(a['pred'], a['real'])[0, 1]) if np.std(a['pred']) > 0 else None,
        sign_agreement=float(np.mean(a['sgn'])), band_coverage_3paths=float(np.mean(a['cov'])),
        median_abs_err_over_sd=float(np.median(a['z'])), rmse_pips=float(np.sqrt(np.mean((np.array(a['pred'])-np.array(a['real']))**2))*1e4),
        real_move_rms_pips=float(np.sqrt(np.mean(np.array(a['real'])**2))*1e4)) for j, a in acc.items()}
out['foresight_calibration'] = cal
dump('G_coverage.json', out)
print(json.dumps({k: v for k, v in out.items() if k not in ('coverage', 'banks')}, default=str)[:6000])
print(json.dumps(out['banks'], default=str)[:1500])
