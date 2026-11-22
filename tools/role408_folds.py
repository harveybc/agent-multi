"""Chronological head refits for a fixed encoder; resumable, no GPU."""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import fcntl
import json
import os

OUT=Path('/var/tmp/role408-managed-v1')
SEEDS=(7,19,43)
FOLDS=(2022,2023)


def one(task):
    import torch
    from tools.role404_fit import fit
    from tools.role404_data import file_sha
    y,s,r=task
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=OUT/str(y);d=root/'champion'/f'cls-{r}-4-{s}'
    if not (d/'RESULT.json').is_file():fit(r,4,s,'cls',paired_root=root,fold_end=y)
    res=json.loads((d/'RESULT.json').read_text())
    assert res['weights']==file_sha(d/'expert.pt') and res['score_year']==y and res['epoch']-res['fork']==20
    return dict(year=y,seed=s,role=r,epoch=res['epoch'])


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    from tools.role404_data import save
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[(y,s,r) for s in SEEDS for y in FOLDS for r in ('CLOSE_BUY','CLOSE_SELL','OPEN_BUY','OPEN_SELL')]
    with (OUT/'folds.lock').open('a') as fd:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
            receipts=list(pool.map(one,tasks))
        save(OUT/'FOLDS_DONE.json',receipts)


if __name__=='__main__':main()
