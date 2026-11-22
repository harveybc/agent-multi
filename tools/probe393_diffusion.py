"""CPU integration probe: one-step joint conditional diffusion, not an approved simulator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, n, c, y):
        super().__init__()
        self.history = nn.GRU(n, 32, batch_first=True)
        self.step = nn.Embedding(32, 12)
        self.noise = nn.Sequential(nn.Linear(32+c+y+12, 96), nn.SiLU(),
                                   nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, y))

    def forward(self, x, controls, y, t):
        _, h = self.history(x)
        return self.noise(torch.cat([h[-1], controls, y, self.step(t)], -1))


def run(src, out, epochs, objective, bridge):
    torch.manual_seed(19)
    torch.set_num_threads(2)
    with np.load(src) as f:
        origin = f["origin"]
        x = torch.from_numpy(f["history_bank"][origin[:, None]+f["offsets"]])
        y = torch.from_numpy(f["targets"]).float()
        ctl = f["controls"][f["segment"]]
        states = np.eye(int(f["controls"][:,0].max())+1)[ctl[:,0].astype(int)]
        parts=[states,ctl[:,1:],f["phase"]]
        if bridge:
            rate=f["remaining"]/f["left"]
            parts.extend([f["remaining"],rate])
        c = torch.tensor(np.column_stack(parts),dtype=torch.float32)
        split = np.datetime64("2022-01-01","ns").astype(np.int64)
        fit = np.flatnonzero(f["clock_ns"][origin+1] < split)
        check = np.flatnonzero(f["clock_ns"][origin]>=split)
    # Signed roots avoid clipping denoiser outputs. Square the sample to
    # recover a nonnegative wick, including true zero wicks.
    observed=y.clone()
    if bridge:
        rate=torch.tensor(rate,dtype=y.dtype)
        # Generate a residual around progress per remaining bar, not a forced
        # endpoint. Pivot targets are extrema, never treated as close prices.
        y[:,1]-=rate
    assert (y[:,2:4]>=0).all()
    y[:,2:4] = y[:,2:4].sqrt() * (torch.randint(2,y[:,2:4].shape)*2-1)
    ym, ys = y[fit].mean(0), y[fit].std(0).clamp_min(1e-10)
    cm, cs = c[fit].mean(0), c[fit].std(0).clamp_min(1e-10)
    y, c = (y-ym)/ys, (c-cm)/cs
    assert all(torch.isfinite(v).all() for v in [x,c,y])
    model = Decoder(x.shape[-1],c.shape[-1],y.shape[-1])
    opt = torch.optim.AdamW(model.parameters(),lr=.001)
    theta = torch.linspace(0,1,33)
    curve = torch.cos((theta+.008)/1.008*torch.pi/2)**2
    beta = (1-curve[1:]/curve[:-1]).clamp(max=.999)
    alpha = (1-beta).cumprod(0)
    rng = torch.Generator().manual_seed(17)
    hold = torch.tensor(check)
    steps = torch.randint(32,(len(check),),generator=rng)
    noise = torch.randn(y[hold].shape,generator=rng)
    corrupt = lambda clean,t,e: alpha[t,None].sqrt()*clean + (1-alpha[t,None]).sqrt()*e
    def target(clean,t,e):
        return e if objective=="eps" else alpha[t,None].sqrt()*e-(1-alpha[t,None]).sqrt()*clean
    noisy = corrupt(y[hold],steps,noise)
    perm = torch.randperm(len(hold),generator=rng)
    history=[]
    for epoch in range(epochs):
        model.train()
        for ix in torch.tensor(fit)[torch.randperm(len(fit))].split(256):
            t = torch.randint(32,(len(ix),))
            e = torch.randn_like(y[ix])
            loss = (model(x[ix],c[ix],corrupt(y[ix],t,e),t)-target(y[ix],t,e)).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            errors=[]
            for hist in (hold,hold[perm]):
                outputs = [model(x[hist[j]],c[hold[j]],noisy[j],steps[j]) for j in
                           torch.arange(len(hold)).split(256)]
                errors.append(float((torch.cat(outputs)-target(y[hold],steps,noise)).square().mean()))
            history.append(dict(epoch=epoch+1,denoising_mse=errors[0],shuffled_history_mse=errors[1]))
    out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(state=model.state_dict(),ym=ym,ys=ys,cm=cm,cs=cs,beta=beta,
                    objective=objective,bridge=bridge,dims=(x.shape[-1],c.shape[-1],y.shape[-1])),out/"PROBE.pt")
    with torch.no_grad():
        ids = hold[perm[:512]]
        g = torch.Generator().manual_seed(31)
        sample = torch.randn((len(ids),y.shape[-1]), generator=g)
        for j in range(31,-1,-1):
            t = torch.full((len(ids),),j)
            pred = model(x[ids],c[ids],sample,t)
            clean = ((sample-(1-alpha[j]).sqrt()*pred)/alpha[j].sqrt() if objective=="eps"
                     else alpha[j].sqrt()*sample-(1-alpha[j]).sqrt()*pred)
            prev = alpha[j-1] if j else torch.tensor(1.)
            sample = (prev.sqrt()*beta[j]*clean + (1-beta[j]).sqrt()*(1-prev)*sample)/(1-alpha[j])
            if j:
                variance=beta[j]*(1-alpha[j-1])/(1-alpha[j])
                sample+=variance.sqrt()*torch.randn(sample.shape,generator=g)
        sample=sample*ys+ym
        if bridge:
            sample[:,1]+=rate[ids]
        sample[:,2:4]=sample[:,2:4].square()
        assert torch.isfinite(sample).all()
        np.save(out/"RAW_ONE_STEP.npy",sample.numpy())
    report=dict(stage="NON_GOVERNING_ONE_STEP", caveat="regime labels and recipe scaler fixed upstream; not blind",
                data_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), fit=len(fit),check=len(check),
                history=history,negative_wick_rows=int((sample[:,2:4]<0).any(1).sum()),
                body_abs_median_bps=float(sample[:,1].abs().median()*10000),
                reference_abs_median_bps=float(observed[ids,1].abs().median()*10000),
                body_abs_p95_bps=float(torch.quantile(sample[:,1].abs(),.95)*10000),
                reference_abs_p95_bps=float(torch.quantile(observed[ids,1].abs(),.95)*10000),
                objective=objective,bridge=bridge,
                generated_rows=len(ids))
    (out/"REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(out,history[-1],report["body_abs_median_bps"],report["reference_abs_median_bps"])


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("src",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--epochs",type=int,required=True)
    p.add_argument("--objective",choices=["eps","velocity"],default="eps")
    p.add_argument("--bridge",action="store_true")
    a=p.parse_args()
    run(a.src,a.out,a.epochs,a.objective,a.bridge)
