"""Run the fixed three-seed classification comparison with CPU-only workers."""
import fcntl
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

ROOT=Path('/var/tmp/role406-cls4-paired')
SEEDS=(7,19,43)
ROLES=('OPEN_BUY','OPEN_SELL','CLOSE_BUY','CLOSE_SELL')


def fit_one(key):
    import torch
    from tools.role404_fit import fit
    from tools.role404_data import file_sha
    role,s=key
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    d=ROOT/'champion'/f'cls-{role}-4-{s}'
    if not (d/'RESULT.json').is_file():fit(role,4,s,'cls',paired_root=ROOT)
    r=json.loads((d/'RESULT.json').read_text())
    if r['weights']!=file_sha(d/'expert.pt'):raise ValueError('champion digest')
    return r['pair_status']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU experiment only')
    from tools.role404_data import save
    from tools.role404_account import run
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results={}
        with ProcessPoolExecutor(max_workers=3,mp_context=get_context('spawn')) as pool:
            for seed in SEEDS:
                keys=[(r,seed) for r in ROLES]
                statuses=list(pool.map(fit_one,keys))
                arms=['champion']+(['fine','explore'] if statuses==['COMPLETE']*4 else [])
                result=dict(roles=dict(zip(ROLES,statuses)),accounts={})
                for arm in arms:
                    run(ROOT/arm,[('cls',4,seed)])
                    p=ROOT/arm/'accounts'/f'cls-4-{seed}.json'
                    result['accounts'][arm]=json.loads(p.read_text())
                results[str(seed)]=result
                save(ROOT/'COMPARISON.json',results)
        save(ROOT/'DONE.json',dict(seeds=list(SEEDS),complete_pairs=sum(
            'explore' in r['accounts'] for r in results.values()),promoted=False))


if __name__=='__main__':main()
