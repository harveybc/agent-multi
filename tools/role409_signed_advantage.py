"""Chronological 408 contrast. Fixed standardized ridge, no validation tuning."""
import os
from pathlib import Path

import numpy as np
import torch
from tools import role408_router as base


def fit(xs, ys):
    x = np.concatenate(xs).astype(float)
    y = np.concatenate(ys)
    assert x.shape == (len(y), 208) and np.isfinite(x).all()
    mu = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.
    z = (x - mu) / scale
    bias = np.empty((2, 4))
    weights = np.empty((208, 2, 4))
    for r, j in np.ndindex(2, 4):
        v = y[:, r, j]
        ok = np.isfinite(v)
        a = z[ok]
        center = a.mean(0)
        a = a - center
        m = v[ok].mean()
        # Fixed penalty 0.1 per observation; do not pick it using evaluation outcomes.
        w = np.linalg.solve(a.T @ a + len(a) * .1 * np.eye(208), a.T @ (v[ok] - m))
        weights[:, r, j] = w
        bias[r, j] = m - center @ w
    return mu, scale, weights, bias, np.nanmean(y, axis=0)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    out = Path('/var/tmp/role409-direct-v1')
    out.mkdir(exist_ok=True)
    bundle = base.ROOT / 'model/61/bundle'
    joined, orig = base.assemble(base.PARENT, bundle, base.ROOT/'history397-61-r2')
    src, _ = base.load_bundle(bundle, 'cpu')
    risk = base.read(base.PARENT/'RISK.json')
    assert base.file_sha(base.DEST/'market.npz') == base.read(base.DEST/'CACHE.json')['sha']
    history = {s: ([], []) for s in base.SEEDS}
    result = {}
    for year in (2022, 2023, 2024):
        m = orig if year == 2024 else joined
        ins = m.inputs
        ix = np.flatnonzero(ins.valid & (ins.row_ts_ns >= base.ns(f'{year}-01-01')) &
                            (ins.row_ts_ns < base.ns(f'{year+1}-01-01')))
        if year < 2024:
            with np.load(base.DEST/'market.npz') as z:
                assert np.array_equal(ins.row_ts_ns, z['t']) and np.array_equal(ins.valid, z['valid'])
                bank = torch.from_numpy(z['x'][ix])
        else:
            with torch.inference_mode(), src.loss.actor_network_params.to_module(src.actor):
                enc = src.actor_net.encoder.eval()
                bank = torch.cat([enc({'features': torch.from_numpy(ins.windows[ix[k:k+256]])}) for k in range(0, len(ix), 256)])
        means = base.class_means(year)
        for seed in base.SEEDS:
            xs, ys = history[seed]
            pn = base.Panel(m, year, seed, bank, ix, means, risk)
            x = bank[pn.index[pn.envs[0].rows[pn.bars]]].numpy()
            path = base.OUT/f'score-{seed}-{year}'
            receipt = base.read(path/'RECEIPT.json')
            assert receipt['spec']['heads'] == pn.hashes and receipt['sha'] == base.file_sha(path/'labels.npz')
            if xs:
                mu, sd, w, b, avg = fit(xs, ys)
                scores = np.tensordot((x.astype(float)-mu)/sd, w, axes=1) + b
                scalar = base.apply_map(pn.p, np.array(base.read(path/'MAP.json')['coefficients']))
                offers = dict(direct=scores, scalar=scalar, mean=np.broadcast_to(avg, scores.shape))
                assert all(np.isfinite(v).all() for v in offers.values())
                accounts = {k: pn.account(v) for k,v in offers.items()}
                assert accounts['scalar'] == base.read(path/'ACCOUNT.json')['cal']
            with np.load(path/'labels.npz') as z:
                assert np.array_equal(z['origin_ns'], pn.e.v[pn.bars]+base.HOUR)
                assert np.array_equal(z['p'], pn.p)
                y = z['y']
            if xs:
                realized = {}
                for k, v in offers.items():
                    d, h = base.decisions(v)
                    i = np.flatnonzero(d)
                    gains = y[i, d[i]-1, h[i]]
                    gains = gains[np.isfinite(gains)]
                    realized[k] = dict(filled_offers=len(gains), mean_bp=float(gains.mean()) if len(gains) else None)
                result[f'{year}-{seed}'] = dict(account=accounts, hypothetical=realized)
                base.save(out/'RESULTS.json', result)
                print(year, seed, accounts['direct']['ret'], accounts['direct']['trades'], flush=True)
            if year < 2024:
                assert receipt['end'] < base.ns(f'{year+1}-01-01')
                xs.append(x)
                ys.append(y)
    base.save(out/'DONE.json', dict(comparisons=len(result), ridge_per_sample=.1))


if __name__ == '__main__':
    main()
