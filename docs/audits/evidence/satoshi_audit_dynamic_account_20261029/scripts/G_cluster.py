import json,sys,numpy as np
sys.path.insert(0,sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs
d=json.load(open(sys.argv[2]))
out={}
rng=np.random.default_rng(20261029)
for split in ('calibration','confirmation'):
    x=np.array(d[f'BOOT|{split}']['diffs']); pairs=x.reshape(6,2).mean(1)
    bs=pairs[rng.integers(0,6,(100000,6))].mean(1)
    out[split]=dict(pair_means=pairs.tolist(),mean=float(pairs.mean()),ci95_pair_cluster=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                    positive_pairs=int((pairs>0).sum()))
# shared innovations across volatility variants and mirror
for seed in (7,43):
    ws=[World(s,CFG) for s in panel_specs(seed)]
    for p in range(3):
        a,bm,c,dm=ws[4*p:4*p+4]
        ia=np.diff(a.env.c); ic=np.diff(c.env.c)
        out[f'seed{seed}_{a.spec["pattern"]}']=dict(mirror_exact=bool(np.allclose(a.env.c+bm.env.c,2.2)),
            same_turn_phase_gap_innovations_across_volatility=bool(np.corrcoef(ia,ic)[0,1]>.5),corr_increments=float(np.corrcoef(ia,ic)[0,1]),
            report_noise_identical_across_patterns=bool(np.allclose(ws[0].report(30).paths-np.column_stack((ws[0].env.o,ws[0].env.h,ws[0].env.l,ws[0].env.c))[31:ws[0].end],
                                                                 ws[4].report(30).paths-np.column_stack((ws[4].env.o,ws[4].env.h,ws[4].env.l,ws[4].env.c))[31:ws[4].end])))
print(json.dumps(out,indent=1))
