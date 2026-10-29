#!/usr/bin/env python3
"""Sequential, resumable TRAIN-only DEAP opportunity search, never a training job."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.zigzag_opportunities import OpportunityEvaluator, engine, load_tape, schedule, sha


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def identity(spec):
    for role in ('input_sha256', 'engine_sha256', 'code_sha256'):
        for key, digest in spec[role].items():
            path = spec['inputs'][key] if role == 'input_sha256' else Path(spec['engine_root'] if role == 'engine_sha256' else ROOT)/key
            if sha(path) != digest:
                raise ValueError('PIN_CHANGED:'+role+':'+key)
    if spec['budget']['population'] < 15 or spec['budget']['generations'] < 30:
        raise ValueError('OWNER_MINIMUM_BUDGET')
    if spec['software'] != {k: importlib.metadata.version(k) for k in spec['software']}:
        raise ValueError('SOFTWARE_CHANGED')
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def prepare(args):
    if args.out.exists():
        raise ValueError('NEW_ROOT_REQUIRED')
    parent = json.loads(args.source_spec.read_text())
    paths = {k:parent['inputs'][k] for k in ('bars','closures','recipe')}
    eng = args.engine_root.resolve()
    spec = dict(schema='zigzag_opportunity_deap.v2', inputs=paths, engine_root=str(eng),
                input_sha256={k:sha(v) for k,v in paths.items()},
                engine_sha256={str(p.relative_to(eng)):sha(p) for p in sorted((eng/'app/short_horizon').glob('*.py'))},
                code_sha256={p:sha(ROOT/p) for p in ['app/zigzag_opportunities.py','tools/zigzag_opportunity_deap.py',
                    'tools/audit_zigzag_opportunity_results.py',
                    'docs/work_plan/118_ZIGZAG_NEXT_EVENTS_CORRECTION.md']},
                software={p:importlib.metadata.version(p) for p in ('numpy','pandas','deap','pyarrow')},
                train=['2020-01-01','2024-01-01'], horizons=[3,6,12,24],
                future_pivots=args.future_pivots,
                endpoint_policy='NEXT_K_EVENTS_NO_TIMED_EXIT',
                budget=dict(population=30,generations=300,seeds=[7,19,43]),
                bounds=dict(depth=[2,96],deviation_pips=[1,120],backstep=[1,24]),
                mode='RETROSPECTIVE_TRAIN_OPPORTUNITY_MAP_NOT_STRATEGY_QUALIFICATION')
    ident = identity(spec)
    tape = load_tape(spec)
    args.out.mkdir(parents=True)
    write(args.out/'SPEC.private.json',spec)
    write(args.out/'IDENTITY.json',dict(identity=ident))
    write(args.out/'DATA_CHECK.json',tape.metadata)
    print(json.dumps(dict(state='PREPARED',identity=ident,**tape.metadata)))


def decode(ind):
    d = min(96,max(2,int(round(ind[0]))))
    return (d,round(min(120,max(1,float(ind[1]))),1), min(d-1,24,max(1,int(round(ind[2])))))


def search(evaluate, seed, population=30, generations=40, progress=None):
    from deap import base, creator, tools
    if population < 15 or generations < 30:
        raise ValueError('OWNER_MINIMUM_BUDGET')
    if not hasattr(creator, 'OpportunityFitness'):
        creator.create('OpportunityFitness', base.Fitness, weights=(1.0,1.0))
        creator.create('OpportunityIndividual', list, fitness=creator.OpportunityFitness)
    state = random.getstate()
    random.seed(seed)
    try:
        def new():
            d=random.randint(2,96)
            return creator.OpportunityIndividual([d,round(random.uniform(1,120),1),random.randint(1,min(24,d-1))])
        pop=[new() for _ in range(population)]
        # Cover shallow scales without assuming they are good: these are initial candidates.
        for i,d in enumerate((2,3,4,6,8,12,18,24,36,48)):
            pop[i][:]=[d,5.0,1]
        history=[]
        def score(ind):
            r=evaluate(decode(ind))
            return (float(r['eligible']),r['fitness'])
        for ind in pop:
            ind.fitness.values=score(ind)
        for g in range(1,generations+1):
            off=[copy.deepcopy(i) for i in tools.selTournament(pop,population-2,tournsize=3)]
            for a,b in zip(off[::2],off[1::2]):
                if random.random()<.8:
                    tools.cxUniform(a,b,indpb=.5)
            for ind in off:
                if random.random()<.4:
                    values=new()
                    for k in range(3):
                        if random.random()<1/3:
                            ind[k]=values[k]
                ind[:]=decode(ind)
                ind.fitness.values=score(ind)
            pop=[copy.deepcopy(i) for i in tools.selBest(pop,2)]+off
            best=tools.selBest(pop,1)[0]
            history.append(dict(generation=g,eligible=bool(best.fitness.values[0]),fitness=best.fitness.values[1],params=decode(best)))
            if progress:
                progress(history[-1])
        best=sorted(pop,key=lambda i:(-i.fitness.values[0],-i.fitness.values[1],decode(i)[0],decode(i)[2],decode(i)[1]))[0]
        return dict(seed=seed,population=population,generations=generations,best=evaluate(decode(best)),history=history)
    finally:
        random.setstate(state)


def authenticated(path, ident):
    r=json.loads(path.read_text())
    if r['identity'] != ident:
        raise ValueError('RESUME_IDENTITY')
    payload=r['payload']
    if r['sha256'] != hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():
        raise ValueError('RESUME_ARTIFACT')
    return payload


def receipt(path, ident, payload):
    write(path,dict(identity=ident,sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload=payload))


def run(out):
    spec=json.loads((out/'SPEC.private.json').read_text())
    ident=identity(spec)
    if json.loads((out/'IDENTITY.json').read_text())['identity'] != ident:
        raise ValueError('SPEC_IDENTITY')
    lock=(out/'QUEUE.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tape=load_tape(spec)
    z,_,_=engine(spec['engine_root'])
    rows=[]; started=time.monotonic()
    for h in spec['horizons']:
        finished=out/f'h{h}_FINAL.json'
        if finished.exists():
            row=authenticated(finished,ident)
            if sha(out/row['records_file']) != row['records_sha256']:
                raise ValueError('FINAL_RECORD_HASH')
            rows.append(row); continue
        if spec.get('endpoint_policy') != 'NEXT_K_EVENTS_NO_TIMED_EXIT':
            raise ValueError('NEW_ENDPOINT_PROTOCOL_REQUIRES_NEW_ROOT')
        ev=OpportunityEvaluator(tape,z,h,future_pivots=spec['future_pivots'])
        cache={}
        ledger=out/f'h{h}_CANDIDATES.jsonl'
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                try:
                    item=json.loads(line)
                except json.JSONDecodeError:
                    continue  # interrupted line; never adopted
                if item['identity'] != ident:
                    raise ValueError('LEDGER_IDENTITY')
                result=item['result']
                if item['sha256'] != hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest():
                    raise ValueError('LEDGER_HASH')
                cache[tuple(item['genes'])]=result
        with ledger.open('a') as log:
            def evaluate(genes):
                if genes not in cache:
                    result=ev.evaluate(genes)
                    cache[genes]=result
                    log.write(json.dumps(dict(identity=ident,genes=genes,result=result,
                        sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()))+'\n')
                    log.flush()
                return cache[genes]
            chains=[]
            for seed in spec['budget']['seeds']:
                ck=out/f'h{h}_seed{seed}.json'
                if ck.exists():
                    chains.append(authenticated(ck,ident)); continue
                tick=time.monotonic()
                def progress(r):
                    p=dict(state='RUNNING',horizon=h,seed=seed,elapsed_seconds=time.monotonic()-started,
                           unit_seconds=time.monotonic()-tick,unique_candidates=len(cache),**r)
                    write(out/'PROGRESS.json',p)
                    if r['generation']%10==0:
                        print(json.dumps(p),flush=True)
                ch=search(evaluate,seed,spec['budget']['population'],spec['budget']['generations'],progress)
                ch['seconds']=time.monotonic()-tick
                receipt(ck,ident,ch); chains.append(ch)
        eligible=[c for c in chains if c['best']['eligible']]
        if not eligible:
            raise ValueError(f'NO_ELIGIBLE_CANDIDATE:h{h}')
        winner=sorted(eligible,key=lambda c:(-c['best']['fitness'],c['best']['params']['depth'],
                      c['best']['params']['backstep'],c['best']['params']['deviation_pips']))[0]
        p=winner['best']['params']; genes=(p['depth'],p['deviation_pips'],p['backstep'])
        detail=ev.evaluate(genes,details=True)
        records=detail.pop('records')
        file=out/f'h{h}_OPPORTUNITIES.npz'
        np.savez_compressed(file,**records)
        # This diagnostic is business-feasible; the oracle objective above is not
        # truncated at flat. Infeasible intervals are omitted, never repriced.
        supported=records['pair_valid']
        orig=records['origin'][supported]
        rel=records['release'][:,supported]
        profits=records['net'][:,supported]
        feasible=rel <= tape.limit[orig][None,:]
        optimal=schedule(orig,np.where(feasible,rel,orig[None,:]+1),
                         np.where(feasible,profits,0.0),len(tape.open))
        receipt(out/f'h{h}_INTERVAL_SCHEDULE.json',ident,optimal)
        trades=optimal['trades']
        sensitivity=[ev.evaluate((genes[0],v,genes[2])) for v in (1.,5.,60.,120.)]
        row=dict(horizon=h,seed=winner['seed'],**detail,records_file=file.name,records_sha256=sha(file),
                 seeds=[dict(seed=c['seed'],fitness=c['best']['fitness'],eligible=c['best']['eligible']) for c in chains],
                 unique_candidates=len(cache),population=spec['budget']['population'],generations=spec['budget']['generations'],
                 deviation_probe=[dict(value=r['params']['deviation_pips'],fitness=r['fitness'],pivots=r['pivots']) for r in sensitivity],
                 interval_net_pips=optimal['net_pips'],interval_trades=len(trades),
                 interval_buy_pct=100*sum(t['side']=='buy' for t in trades)/len(trades) if trades else None,
                 interval_win_pct=100*sum(t['net_pips']>0 for t in trades)/len(trades) if trades else None)
        receipt(finished,ident,row); rows.append(row)
        print(json.dumps(dict(state='HORIZON_COMPLETE',horizon=h,params=p,fitness=row['fitness'])),flush=True)
    identity(spec)
    receipt(out/'RESULTS.json',ident,dict(rows=rows,data=tape.metadata,elapsed_seconds=time.monotonic()-started,
            scope=spec['mode'],financial_model_runs=0,validation_read=False,test_read=False))
    write(out/'PROGRESS.json',dict(state='COMPLETE',horizons=len(rows),identity=ident))
    lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['prepare','run'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-spec',type=Path)
    p.add_argument('--engine-root',type=Path)
    p.add_argument('--future-pivots',type=int,choices=[2,3],default=3)
    a=p.parse_args()
    if a.command=='prepare':
        prepare(a)
    else:
        run(a.out)
