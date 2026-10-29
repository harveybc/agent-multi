#!/usr/bin/env python3
"""Pinned sequential NSGA-II on synthetic native accounts; never trains a model."""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import deap
from deap import base, creator, tools as D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dynamic_account_arbiter import bounds, decode, frontier, GENES, HORIZONS
from app.dynamic_account_panel import World, panel_specs, evaluate, BASE
from app.zigzag_opportunities import engine, sha
from tools.verify_absolute_early_close import source_pins, write


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class Receipts:
    """Append/fsync each result; reject truncation or altered records, not silently refit."""
    def __init__(self,path,identity):
        self.path=path;self.identity=identity;self.rows={};self.chain=identity
        if path.exists():
            for line in path.read_text().splitlines():
                r=json.loads(line);seal=r.pop('receipt')
                if r['previous']!=self.chain or r['identity']!=identity or digest(r)!=seal:
                    raise ValueError('CANDIDATE_RECEIPT_IDENTITY')
                if r['key']!=digest(r['genes']) or r['key'] in self.rows:
                    raise ValueError('CANDIDATE_KEY')
                self.chain=seal;self.rows[r['key']]=r['result']

    def append(self,genes,result):
        key=digest(genes)
        if key in self.rows: raise ValueError('DUPLICATE_CANDIDATE_RECEIPT')
        body=dict(identity=self.identity,previous=self.chain,key=key,genes=genes,result=result)
        seal=digest(body)
        with self.path.open('a') as f:
            f.write(json.dumps(body|dict(receipt=seal),sort_keys=True,allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        self.rows[key]=result;self.chain=seal


def compact(result):
    rows=result['worlds'];n=sum(r['orders'] for r in rows);buy=sum(r['buys'] for r in rows)
    return dict(objectives=result['objectives'],eligible=result['eligible'],mean_orders=result['mean_orders'],
        orders=n,buys=buy,sells=n-buy,wins=sum(r['wins'] for r in rows),
        buy_wins=sum(r['buy_wins'] for r in rows),sell_wins=sum(r['sell_wins'] for r in rows),
        early_closes=sum(r['closes'].get('AGENT_CLOSE',0) for r in rows))


def controls():
    all_pairs={(h,d) for h in HORIZONS for d in (1,-1)}
    guarded=list(BASE);guarded[5]=1.;guarded[6]=.1;guarded[7]=.05;guarded[9]=.02;guarded[10]=2
    rows=[('nop',BASE,set()),('equal_guarded',guarded,all_pairs),('greedy',BASE,all_pairs)]
    rows.extend((f'only_{h}',BASE,{(h,d) for d in (1,-1)}) for h in HORIZONS)
    return rows


def sources(root):
    out=source_pins(root)
    for name in ('app/close_hold_controller.py','app/dynamic_account_arbiter.py',
                 'app/dynamic_account_panel.py','tools/optimize_dynamic_account.py'):
        out['agent_multi'][name]=sha(ROOT/name)
    return out


def write_summary(output,report):
    lines=['# Synthetic dynamic account Pareto experiment','',
        f"Status: {report['status']}. Identity: `{report['identity']}`.",
        'This is simulated model output, not financial validation or a learned policy.',
        'The frontier is empirical among evaluated policies, not a global optimum.',
        '', '| Policy | Panel | Return % | Worst DD % | Orders/week | Buy / sell | Win % | Early closes |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in report['summaries']:
        win=100*r['wins']/r['orders'] if r['orders'] else 0.
        lines.append(f"| {r['name'][:12]} | {r['split']} | {r['objectives'][0]:.4f} | {-r['objectives'][1]:.4f} | "
            f"{r['mean_orders']:.2f} | {r['buys']} / {r['sells']} | {win:.2f} | {r['early_closes']} |")
    lines.extend(['',f"Unique candidates: {report['unique_candidates']}; nondominated policies: {report['frontier_size']}.",
        'Return and drawdown are separate objectives. Confirmation never chooses the policy.',
        'REPLAYS.json.gz holds native ledgers, close reasons, horizon allocation, costs and Sharpe.',
        'REPORT.json holds finite-game matrices, unilateral deviations and actual variation of every gene.',
        'SL/TP geometry is fixed to the shared median profiles in this allocation experiment.',
        'Closing at the local position optimum and whole-account opportunity allocation are distinct objectives.',
        'No claim of exhaustive coverage of every market condition, of a qualified reporter, or of deployment readiness.'])
    (output/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    (output/'AUDIT_REQUEST.md').write_text('\n'.join([
        '# Satoshi: dynamic account mechanism audit', '',
        f"Exact identity: `{report['identity']}`. Status: {report['status']}.",
        'Review the complete MANIFEST, REPORT, FRONTIER, CANDIDATES receipts and REPLAYS before a verdict.',
        '1. Independently reproduce native cash, drawdown, funding, time gates and one-position transitions.',
        '2. Attack the per-unit cache and compare it with the uncached position-conditioned report route.',
        '3. Check full-account opportunity comparisons; no overlapping hypothetical PnL enters account payoff.',
        '4. Recompute the empirical frontier, activity constraint and fixed confirmation representative rule.',
        '5. Check that controls may dominate optimized policies; report such cases explicitly.',
        '6. Inspect fixed-horizon risk profiles versus actual duration, mirrored paths and direction behavior.',
        '7. Distinguish on/off unilateral diagnostics from global Nash and Pareto certificates.',
        '8. Attack the assumed noisy/ideal reports and scenario coverage; neither is evidence of learnability.',
        '9. Check early-close counterexamples at account level: local position-optimal is not globally optimal.',
        'No financial learning or sequential floating selection is authorized by this report.',
        'For a smoke run this request is a generated template only, not a final audit submission.', '']))


def search(worlds,receipts,seed,population,generations,output,identity,check):
    if not hasattr(creator,'AccountFitness'):
        creator.create('AccountFitness',base.Fitness,weights=(1.,1.))
        creator.create('AccountIndividual',list,fitness=creator.AccountFitness)
    random.seed(seed);lo,hi=zip(*bounds());evaluations=0;history=[];started=time.monotonic()
    def fit(ind):
        nonlocal evaluations
        ind[:]=decode(ind);key=digest(ind)
        if key not in receipts.rows:
            result=compact(evaluate(worlds,ind));receipts.append(list(ind),result);evaluations+=1
        r=receipts.rows[key]
        return tuple(r['objectives']) if r['eligible'] else (-1e6,-1e6)
    pop=[creator.AccountIndividual(BASE)]
    pop.extend(creator.AccountIndividual([random.uniform(a,b) for a,b in bounds()]) for _ in range(population-1))
    for x in pop: x.fitness.values=fit(x)
    pop=D.selNSGA2(pop,population)
    for generation in range(1,generations+1):
        children=[copy.deepcopy(x) for x in D.selTournamentDCD(pop,population)]
        for a,b in zip(children[::2],children[1::2]):
            if random.random()<.9: D.cxSimulatedBinaryBounded(a,b,eta=15,low=lo,up=hi)
            for x in (a,b):
                D.mutPolynomialBounded(x,eta=20,low=lo,up=hi,indpb=1/len(GENES))
                x.fitness.values=fit(x)
        pop=D.selNSGA2(pop+children,population)
        history.append(dict(generation=generation,genes=[list(x) for x in pop],
                            fitness=[list(x.fitness.values) for x in pop]))
        if generation%10==0 or generation==generations:
            check()
            state=dict(status='RUNNING',phase='DEAP',seed=seed,generation=generation,
                generations=generations,unique_candidates=len(receipts.rows),new_this_seed=evaluations,
                seed_elapsed_seconds=time.monotonic()-started,identity=identity)
            write(output/'STATE.json',state);print(json.dumps(state),flush=True)
    result=dict(seed=seed,population=population,generations=generations,history=history,
                new_evaluations=evaluations,identity=identity)
    write(output/f'SEED_{seed}.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true',help='two generations, NEVER a scientific result')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'LOCK').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine_root=a.engine_root.resolve();engine(engine_root)
    from app.short_horizon.expert_guided import absolute
    if Path(absolute.__file__).resolve()!=engine_root/'app/short_horizon/expert_guided/absolute.py':
        raise ValueError('WRONG_NATIVE_ENGINE')
    config_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    config=json.loads(config_path.read_text());code=sources(engine_root)
    generations=2 if a.smoke else 300;seeds=[7] if a.smoke else [7,19,43]
    manifest=dict(schema='dynamic_account_pareto.v1',synthetic_only=True,smoke=a.smoke,
        population=20,generations=generations,seeds=seeds,genes=GENES,bounds=bounds(),
        calibration=panel_specs(7),confirmation=panel_specs(43),sources=code,
        config_sha256=sha(config_path),versions=dict(deap=deap.__version__,numpy=np.__version__,pandas=pd.__version__),
        algorithm=dict(selection='NSGA-II',crossover_eta=15,crossover_probability=.9,
                       mutation_eta=20,mutation_gene_probability=1/11),
        objectives=['mean_account_return_pct','negative_worst_account_drawdown_pct'],
        constraint='mean_orders_per_120bar_market_week >= 3',
        controls=[dict(name=n,genes=list(g),enabled=sorted(e)) for n,g,e in controls()],
        confirmation_rule='up to five evenly spaced return-rank frontier representatives; no retuning',
        scalar_nash_diagnostic='return_pct - worst_drawdown_pct; only eight on/off deviations')
    identity=digest(manifest);manifest['identity']=identity
    path=a.output/'MANIFEST.json'
    if path.exists() and digest(json.loads(path.read_text()))!=digest(manifest):
        raise ValueError('RESUME_IDENTITY')
    write(path,manifest)
    def check():
        if sources(engine_root)!=code or sha(config_path)!=manifest['config_sha256']:
            raise ValueError('SOURCE_MUTATION')
    started=time.monotonic()
    try:
        receipts=Receipts(a.output/'CANDIDATES.jsonl',identity)
        worlds=[World(s,config) for s in panel_specs(7)]
        for seed in seeds: search(worlds,receipts,seed,20,generations,a.output,identity,check)
        check()
        vectors={k:r['objectives'] for k,r in receipts.rows.items() if r['eligible']}
        selected=frontier(vectors)
        if not selected: raise ValueError('NO_ELIGIBLE_FRONTIER')
        genes={}
        for line in (a.output/'CANDIDATES.jsonl').read_text().splitlines():
            r=json.loads(line);genes[r['key']]=r['genes']
        ranks=sorted(set(int(round(x)) for x in np.linspace(0,len(selected)-1,min(5,len(selected)))))
        representatives=[selected[i] for i in ranks]
        write(a.output/'FRONTIER.json',dict(identity=identity,keys=selected,
            representatives=representatives,rows=[dict(key=k,genes=genes[k],**receipts.rows[k]) for k in selected]))
        write(a.output/'STATE.json',dict(status='RUNNING',phase='FROZEN_CONFIRMATION',identity=identity))
        confirmation=[World(s,config) for s in panel_specs(43)]
        detailed={};summary=[]
        candidates=[(name,g,enabled) for name,g,enabled in controls()]
        candidates.extend((key,genes[key],None) for key in representatives)
        for name,g,enabled in candidates:
            detailed[name]={}
            for split,ws in (('calibration',worlds),('confirmation',confirmation)):
                r=evaluate(ws,g,enabled,verify=True,details=True)
                detailed[name][split]=r;summary.append(dict(name=name,split=split,**compact(r)))
                if name in representatives and split=='calibration' and compact(r)!=receipts.rows[name]:
                    raise ValueError('FINALIST_REPLAY_MISMATCH')
        # Diagnostic only: fixed middle-return frontier point, never confirmation-selected.
        key=representatives[len(representatives)//2];g=genes[key]
        full=evaluate(worlds,g);phi=sum(full['objectives']);all_pairs={(h,d) for h in HORIZONS for d in (1,-1)}
        deviations=[]
        for pair in sorted(all_pairs):
            r=evaluate(worlds,g,all_pairs-{pair})
            deviations.append(dict(disabled=pair,objectives=r['objectives'],
                                   eligible=r['eligible'],unilateral_gain=sum(r['objectives'])-phi))
        matrices=[]
        for h in HORIZONS:
            outcomes={}
            for buy in (0,1):
                for sell in (0,1):
                    enabled=({(h,1)} if buy else set())|({(h,-1)} if sell else set())
                    r=evaluate(worlds,g,enabled);outcomes[buy,sell]=r['objectives']
            cells=[]
            for (buy,sell),v in outcomes.items():
                total=sum(v)
                cells.append(dict(buy=buy,sell=sell,account_objectives=v,
                    difference_payoffs=[total-sum(outcomes[0,sell]) if buy else 0.,
                                        total-sum(outcomes[buy,0]) if sell else 0.]))
            matrices.append(dict(horizon=h,cells=cells))
        raw=json.dumps(detailed,sort_keys=True,allow_nan=False).encode()
        (a.output/'REPLAYS.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        check()
        report=dict(status='SMOKE_ONLY' if a.smoke else 'COMPLETE_EMPIRICAL_SYNTHETIC_FRONTIER',
            identity=identity,unique_candidates=len(receipts.rows),frontier_size=len(selected),
            summaries=summary,diagnostic_representative=key,unilateral_deviations=deviations,
            epsilon_on_off=max(0.,max(r['unilateral_gain'] for r in deviations)),payoff_matrices=matrices,
            epsilon_activity_feasible=max([0.]+[r['unilateral_gain'] for r in deviations if r['eligible']]),
            gene_variation={n:dict(min=min(g[i] for g in genes.values()),max=max(g[i] for g in genes.values()),
                                   unique=len({g[i] for g in genes.values()})) for i,n in enumerate(GENES)},
            replays_sha256=sha(a.output/'REPLAYS.json.gz'),receipts_sha256=sha(a.output/'CANDIDATES.jsonl'),
            receipt_tail=receipts.chain,frontier_sha256=sha(a.output/'FRONTIER.json'),
            elapsed_seconds=time.monotonic()-started,global_optimum_certified=False,learned_policy_tested=False)
        write(a.output/'REPORT.json',report)
        write_summary(a.output,report)
        write(a.output/'STATE.json',dict(status='COMPLETE',identity=identity,
            unique_candidates=len(receipts.rows),elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(status=report['status'],frontier_size=len(selected),elapsed=report['elapsed_seconds'])),flush=True)
    except Exception as exc:
        write(a.output/'STATE.json',dict(status='FAILED',identity=identity,error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__': main()
