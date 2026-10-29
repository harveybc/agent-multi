#!/usr/bin/env python3
"""Controlled generation-budget extension followed by audit and paired comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.zigzag_opportunities import sha
from tools.zigzag_opportunity_deap import identity, authenticated, write, run, receipt
from tools.audit_zigzag_opportunity_results import audit


def paired(old, new):
    if old['seed'] != new['seed'] or old['population'] != new['population']:
        raise ValueError('PAIRED_SEARCH_SETTINGS')
    if new['generations'] != 300 or new['population'] < 20:
        raise ValueError('EXTENSION_BUDGET')
    if old['history'] != new['history'][:old['generations']]:
        raise ValueError('GENERATION_PREFIX_MISMATCH')
    if not new['best']['eligible'] or new['best']['fitness'] < old['best']['fitness']:
        raise ValueError('ELITISM_REGRESSION')
    a,b=old['best']['fitness'],new['best']['fitness']
    return dict(seed=old['seed'],baseline_fitness=a,extended_fitness=b,delta=b-a,
                relative_improvement_pct=100*(b/a-1) if a else None,
                prefix_generations_verified=old['generations'],
                baseline_params=old['best']['params'],extended_params=new['best']['params'])


def prepare(baseline, out):
    old=json.loads((baseline/'SPEC.private.json').read_text())
    old_id=identity(old)
    if json.loads((baseline/'IDENTITY.json').read_text())['identity'] != old_id:
        raise ValueError('BASELINE_SPEC_IDENTITY')
    authenticated(baseline/'RESULTS.json',old_id)
    if old['budget'] != dict(population=30,generations=40,seeds=[7,19,43]):
        raise ValueError('EXPECTED_40_GENERATION_BASELINE')
    protected={str(p.relative_to(baseline)):sha(p) for p in sorted(baseline.iterdir()) if p.is_file()}
    spec=json.loads(json.dumps(old))
    spec['budget']['generations']=300
    spec['extension']=dict(baseline_identity=old_id,baseline_file_sha256=protected)
    for path in ('tools/extend_zigzag_opportunity.py','tools/audit_zigzag_opportunity_results.py',
                 'docs/work_plan/116_ZIGZAG_300_GENERATION_CONTRAST.md'):
        spec['code_sha256'][path]=sha(ROOT/path)
    ident=identity(spec)
    if out.exists():
        if json.loads((out/'SPEC.private.json').read_text()) != spec:
            raise ValueError('EXTENSION_RESUME_SPEC')
        return spec,ident
    out.mkdir(parents=True)
    write(out/'SPEC.private.json',spec)
    write(out/'IDENTITY.json',dict(identity=ident))
    adopted={}
    for h in old['horizons']:
        for seed in old['budget']['seeds']:
            authenticated(baseline/f'h{h}_seed{seed}.json',old_id)
        items=[]
        for line in (baseline/f'h{h}_CANDIDATES.jsonl').read_text().splitlines():
            item=json.loads(line)
            digest=hashlib.sha256(json.dumps(item['result'],sort_keys=True).encode()).hexdigest()
            if item['identity'] != old_id or item['sha256'] != digest:
                raise ValueError('BASELINE_LEDGER')
            item['identity']=ident
            items.append(item)
        with (out/f'h{h}_CANDIDATES.jsonl').open('x') as f:
            for item in items:
                f.write(json.dumps(item)+'\n')
        adopted[str(h)]=len(items)
    write(out/'CACHE_ADOPTION.json',dict(baseline_identity=old_id,counts=adopted))
    return spec,ident


def compare(baseline, out, spec, ident):
    old_id=spec['extension']['baseline_identity']
    for path,digest in spec['extension']['baseline_file_sha256'].items():
        if sha(baseline/path) != digest:
            raise ValueError('BASELINE_CHANGED:'+path)
    before=authenticated(baseline/'RESULTS.json',old_id)
    after=authenticated(out/'RESULTS.json',ident)
    if before['data'] != after['data']:
        raise ValueError('DATA_SUPPORT_CHANGED')
    rows=[]
    for a,b in zip(before['rows'],after['rows']):
        if a['horizon']!=b['horizon']:
            raise ValueError('HORIZON_ORDER')
        h=a['horizon']
        seeds=[paired(authenticated(baseline/f'h{h}_seed{s}.json',old_id),
                      authenticated(out/f'h{h}_seed{s}.json',ident)) for s in spec['budget']['seeds']]
        rows.append(dict(horizon=h,baseline_fitness=a['fitness'],extended_fitness=b['fitness'],
                    delta=b['fitness']-a['fitness'],relative_improvement_pct=100*(b['fitness']/a['fitness']-1),
                    baseline_params=a['params'],extended_params=b['params'],
                    median_pivot_hours=b['median_pivot_hours'],seeds=seeds))
    result=dict(state='COMPLETE_PAIRED',baseline_identity=old_id,extended_identity=ident,
                baseline_unchanged=True,prefix_generations_verified=480,rows=rows,
                baseline_seconds=before['elapsed_seconds'],extended_seconds=after['elapsed_seconds'])
    receipt(out/'COMPARISON.json',ident,result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    spec,ident=prepare(a.baseline,a.out)
    print(json.dumps(dict(state='EXTENSION_PREPARED',identity=ident,budget=spec['budget'])),flush=True)
    run(a.out)
    write(a.out/'AUDIT.json',audit(a.out))
    print(json.dumps(compare(a.baseline,a.out,spec,ident),indent=2),flush=True)


if __name__=='__main__':
    main()
