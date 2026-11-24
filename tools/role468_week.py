"""Incremental weekly challenger. Use one newly matured batch stream, never select on its week."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from app.structural_refit import load_bundle


def main(a):
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.src/'dataset_sdelta.npz')
    w = np.load(a.src/'windows.npz')
    np.testing.assert_array_equal(w['ts'], d['ts'])
    ix = np.flatnonzero(d['holdout'])
    ts = d['ts'][ix]
    assert np.all(np.diff(ts) > 0)
    reference = np.load(a.fixed)
    np.testing.assert_array_equal(ts, reference['ts'])
    x = torch.from_numpy(w['windows'][ix]).to(a.device)
    horizon = 6 if a.task == 'short' else 48
    y = d['y_'+a.task][ix]
    ok = d['ok_'+a.task][ix]
    end = ts + horizon * 3600 * 10**9
    start = np.datetime64('2024-01-01', 'ns').astype('int64')  # Monday UTC
    week = 7*24*3600*10**9
    slots = (ts-start)//week
    assert slots.min() >= 0
    ck = torch.load(a.src/f'retsu-phase1-sdelta-tuned/jobs/tuned_{a.task}/champion.pt',
                    weights_only=False, map_location='cpu')
    rec = ck['record']
    mean, std = np.array(rec['target_mean']), np.array(rec['target_std'])
    targets = torch.from_numpy(((y-mean)/std).astype('float32')).to(a.device)
    model, _ = load_bundle(a.bundle, a.device)
    params = dict(model.loss.named_parameters())
    for v in params.values():
        v.requires_grad_(False)
    chosen = []
    with torch.no_grad():
        for name, val in ck['encoder']:
            params[name].copy_(val)
            params[name].requires_grad_(True)
            chosen.append(params[name])
    enc = model.actor_net.encoder
    head = torch.nn.Linear(208, y.shape[1]).to(a.device)
    head.load_state_dict(ck['head'])
    opt = torch.optim.AdamW([*head.parameters(), *chosen], lr=ck['lr'], weight_decay=.004)
    def forward(ii):
        with model.loss.actor_network_params.to_module(model.actor):
            return head(enc({'features': x[ii]})/rec['embedding_scale'])
    with torch.no_grad():
        enc.eval()
        fixed = torch.cat([forward(ii) for ii in np.array_split(np.arange(len(ix)), 20)]).cpu().numpy()*std+mean
    original = reference['tuned'][:, :6] if horizon==6 else reference['tuned'][:, 6:]
    np.testing.assert_allclose(fixed, original, rtol=0, atol=1e-6)
    out = np.full_like(fixed, np.nan)
    reports = []
    last = start
    for slot in np.unique(slots):
        cut = int(start+slot*week)
        train = np.flatnonzero(ok & (end>=last) & (end<cut))
        pred = np.flatnonzero(slots==slot)
        assert not len(train) or end[train].max() < cut <= ts[pred].min()
        enc.train()
        for i in range(0,len(train),256):
            ii = train[i:i+256]
            opt.zero_grad(set_to_none=True)
            loss = (forward(ii)-targets[ii]).abs().mean()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        enc.eval()
        with torch.no_grad():
            out[pred] = forward(pred).cpu().numpy()*std+mean
        state = {n:params[n].detach().cpu() for n,_ in ck['encoder']}
        state.update({'head.'+n:v.detach().cpu() for n,v in head.state_dict().items()})
        snap = a.out/f'{a.task}-{slot:02d}.pt'
        torch.save(state, snap)
        reports.append(dict(week=int(slot),cut_ns=cut, latest_label_ns=int(end[train].max()) if len(train) else None,
                            n_train=len(train),n_pred=len(pred), sha256=hashlib.sha256(snap.read_bytes()).hexdigest()))
        last=cut
    assert np.isfinite(out).all()
    np.savez_compressed(a.out/f'{a.task}.npz',ts=ts,weekly=out,fixed=fixed)
    (a.out/f'{a.task}.json').write_text(json.dumps(dict(lr=ck['lr'],max_fixed_error=float(np.max(abs(fixed-original))),weeks=reports),indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('src','bundle','fixed','out'):
        p.add_argument(k,type=Path)
    p.add_argument('task',choices=['short','long'])
    p.add_argument('--device',default='cuda')
    main(p.parse_args())
