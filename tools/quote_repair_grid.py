"""Resumable full-episode engineering grid. No optimization or financial release."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gzip
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import shutil
import time

from app.frozen_recipe_inputs import require,file_sha
from app.rl_subset import digest
from tools.recipe_sac_queue import read,write,source_pins,software,now

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=None


def cases(config):
    require(config['schema']=='quote_repair_grid.v1','QUOTE_GRID_SCHEMA')
    axes=config['axes'];keys=tuple(axes)
    expected={'drift_before','drift_after','vol_before','vol_after','switch','transition','overlay',
              'noise','shock','seed','mirror','horizon','spread_factor','mode'}
    require(set(keys)==expected and all(isinstance(v,list) and v for v in axes.values()),'QUOTE_GRID_AXES')
    require(all(len({json.dumps(x) for x in v})==len(v) for v in axes.values()),'QUOTE_GRID_DUPLICATE')
    for i,values in enumerate(itertools.product(*(axes[k] for k in keys))):
        yield i,dict(zip(keys,values))


def initialize(recipe,engine):
    import torch
    from tools.systematic_strategy_matrix import setup
    global RUNTIME
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(not torch.cuda.is_available(),'CPU_ONLY')
    rec,cfg=setup(recipe,engine)
    RUNTIME=(cfg,(rec['tensor']['window']['observations'],rec['list']['n']))


def batch(task):
    from app.strategy_repair_episode import run_episode
    index,items,genes,minimum_gain_risk=task
    cfg,shape=RUNTIME;rows=[];started=time.monotonic()
    for i,cell in items:
        spec={k:v for k,v in cell.items() if k not in ('mode','horizon','spread_factor')}
        try:
            result=run_episode(spec,cfg,shape,genes,cell['horizon'],cell['mode'],cell['spread_factor'],minimum_gain_risk)
            rows.append(dict(case_id=i,state='PASS',result=result))
        except Exception as exc:
            rows.append(dict(case_id=i,state='FAIL',cell=cell,error_type=type(exc).__name__,error=str(exc)))
    return dict(index=index,rows=rows,seconds=time.monotonic()-started)


def run(a):
    config=read(a.design)
    total=math.prod(len(v) for v in config['axes'].values())
    require(1<=a.workers<=6 and 1<=a.batch_size<=128,'QUOTE_GRID_WORKERS')
    require(a.limit is None or 0<a.limit<=total,'QUOTE_GRID_LIMIT')
    require(a.sample is None or (a.limit is None and 0<a.sample<=total),'QUOTE_GRID_SAMPLE')
    require(a.storage_reserve_gib>0 and (a.limit or a.sample or a.storage_reserve_gib>=20),'QUOTE_GRID_RESERVE_POLICY')
    sample_ids=set(random.Random(20261027).sample(range(total),a.sample)) if a.sample else None
    # Smoke identity cannot be resumed as complete coverage.
    identity=dict(schema='quote_repair_execution.v1',design=config,limit=a.limit,sample=a.sample,batch_size=a.batch_size,
        recipe_sha256=file_sha(a.recipe),agent_sources=source_pins(ROOT,('app','agent_plugins','env_plugins','tools')),
        engine_sources=source_pins(a.engine_root,('app',)),software=software(),financial_admission=False,
        target_cases=a.sample or a.limit or total,scope=config['scope'],
        storage_reserve_gib=a.storage_reserve_gib,
        effective_business={**read(a.recipe)['business']['evaluation_and_policy'],
                            'execution_price_mode':'bid_ask_v3'},
        per_case_overrides=['profile spread_schedule and weekly_open_spread multiplied by spread_factor'])
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'RUN.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (a.output/'IDENTITY.json').exists():require(read(a.output/'IDENTITY.json')==identity,'QUOTE_GRID_IDENTITY')
        else:write(a.output/'IDENTITY.json',identity)
        if (a.output/'CLOSE.json').exists():
            closed=read(a.output/'CLOSE.json');count=0
            require(closed['identity']==digest(identity) and closed['failures']==0,'QUOTE_GRID_CLOSE')
            for name,sha in closed['receipts'].items():
                receipt=a.output/'batches'/name
                require(receipt.resolve().is_relative_to((a.output/'batches').resolve()) and
                        file_sha(receipt)==sha,'QUOTE_GRID_CLOSED_RECEIPT')
                r=read(receipt)
                require(r['identity']==digest(identity) and r['failures']==0 and
                        file_sha(receipt.with_suffix('.json.gz'))==r['sha256'],'QUOTE_GRID_CLOSED_DATA')
                count+=r['count']
            require(count==closed['completed']==identity['target_cases'],'QUOTE_GRID_CLOSED_COUNT')
            return
        completed=0;failures=0;elapsed=0.;receipts={}
        iterator=iter((row for row in cases(config) if row[0] in sample_ids) if sample_ids is not None
                      else itertools.islice(cases(config),identity['target_cases']))
        index=0
        def remaining():
            nonlocal index,completed,failures,elapsed
            while True:
                items=list(itertools.islice(iterator,a.batch_size))
                if not items:return
                number=index;index+=1
                receipt=a.output/'batches'/f'{number:07d}.json'
                if receipt.exists():
                    r=read(receipt);data=receipt.with_suffix('.json.gz')
                    require(r['identity']==digest(identity) and file_sha(data)==r['sha256'],'QUOTE_GRID_RECEIPT')
                    require(r['first']==items[0][0] and r['last']==items[-1][0] and r['count']==len(items),'QUOTE_GRID_RANGE')
                    completed+=r['count'];failures+=r['failures'];elapsed+=r['seconds'];receipts[receipt.name]=file_sha(receipt)
                    require(r['failures']==0,'QUOTE_GRID_PRIOR_FAILURE_REQUIRES_NEW_VERSION')
                    continue
                yield number,items,config['genes'],config['minimum_gain_risk']
        work=iter(remaining())
        started=time.monotonic()
        def status(state):
            per_case=elapsed/completed if completed else None
            write(a.output/'STATUS.json',dict(state=state,completed=completed,total=identity['target_cases'],failures=failures,
                summed_worker_seconds=elapsed,wall_seconds=time.monotonic()-started,
                estimated_remaining_seconds=(identity['target_cases']-completed)*per_case/a.workers if per_case else None,
                workers=a.workers,financial_admission=False,at=now()))
        status('STARTING')
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize,initargs=(a.recipe,a.engine_root)) as pool:
            active={}
            for _ in range(a.workers):
                task=next(work,None)
                if task is not None:active[pool.submit(batch,task)]=task
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for f in done:
                    task=active.pop(f);value=f.result()
                    require(value['index']==task[0] and [r['case_id'] for r in value['rows']]==[i for i,c in task[1]],'QUOTE_BATCH_IDS')
                    require(shutil.disk_usage(a.output).free>a.storage_reserve_gib*1024**3,'QUOTE_GRID_STORAGE_RESERVE')
                    path=a.output/'batches'/f'{value["index"]:07d}.json.gz';path.parent.mkdir(exist_ok=True)
                    packed=gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),mtime=0)
                    tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as stream:
                        stream.write(packed);stream.flush();os.fsync(stream.fileno())
                    tmp.replace(path)
                    nfail=sum(r['state']=='FAIL' for r in value['rows'])
                    r=dict(identity=digest(identity),sha256=file_sha(path),first=task[1][0][0],last=task[1][-1][0],
                        count=len(value['rows']),failures=nfail,seconds=value['seconds'])
                    receipt=path.with_suffix('');write(receipt,r);receipts[receipt.name]=file_sha(receipt)
                    completed+=r['count'];failures+=nfail;elapsed+=r['seconds']
                    status('FAILED' if failures else 'RUNNING')
                    if failures:
                        for pending in active:pending.cancel()
                        raise RuntimeError('QUOTE_GRID_BEHAVIOR_FAILURE_SEE_RECEIPTS')
                    task=next(work,None)
                    if task is not None:active[pool.submit(batch,task)]=task
        require(completed==identity['target_cases'] and not failures,'QUOTE_GRID_COMPLETE')
        state='ENGINEERING_SMOKE_COMPLETE' if (a.limit or a.sample) else 'ENGINEERING_GRID_COMPLETE_NOT_STRATEGY_RELEASE'
        write(a.output/'CLOSE.json',dict(state=state,identity=digest(identity),completed=completed,failures=failures,
            receipts=receipts,financial_admission=False,learning_qualified=False,at=now()))
        status(state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','engine-root','design','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--limit',type=int)
    p.add_argument('--sample',type=int)
    p.add_argument('--storage-reserve-gib',type=float,default=20.)
    run(p.parse_args())
