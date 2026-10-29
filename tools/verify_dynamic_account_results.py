#!/usr/bin/env python3
"""Verify closed account-search receipts and native replays; optionally export evidence."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dynamic_account_panel import World, evaluate, panel_specs
from app.zigzag_opportunities import engine, sha
from tools.optimize_dynamic_account import Receipts, compact, controls, digest, sources
from tools.verify_absolute_early_close import write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--export',type=Path,required=True)
    a=p.parse_args()
    if a.export.exists(): raise ValueError('NEW_EXPORT_REQUIRED')
    root=a.run;manifest=json.loads((root/'MANIFEST.json').read_text())
    identity=manifest.pop('identity')
    if digest(manifest)!=identity or manifest['smoke']: raise ValueError('FINAL_RUN_IDENTITY')
    report=json.loads((root/'REPORT.json').read_text());front=json.loads((root/'FRONTIER.json').read_text())
    if report['identity']!=identity or front['identity']!=identity: raise ValueError('REPORT_IDENTITY')
    if report['status']!='COMPLETE_EMPIRICAL_SYNTHETIC_FRONTIER': raise ValueError('NOT_COMPLETE')
    for name,field in [('CANDIDATES.jsonl','receipts_sha256'),('REPLAYS.json.gz','replays_sha256'),
                       ('FRONTIER.json','frontier_sha256')]:
        if sha(root/name)!=report[field]: raise ValueError('ARTIFACT_DIGEST')
    receipts=Receipts(root/'CANDIDATES.jsonl',identity)
    if receipts.chain!=report['receipt_tail'] or len(receipts.rows)!=report['unique_candidates']:
        raise ValueError('RECEIPT_CHAIN')
    for seed in manifest['seeds']:
        s=json.loads((root/f'SEED_{seed}.json').read_text())
        if (s['identity']!=identity or s['population']!=20 or s['generations']!=300
                or [x['generation'] for x in s['history']]!=list(range(1,301))):
            raise ValueError('FULL_BUDGET')
        for generation in s['history']:
            if len(generation['genes'])!=20 or len(generation['fitness'])!=20: raise ValueError('POPULATION')
            for genes,fit in zip(generation['genes'],generation['fitness']):
                row=receipts.rows[digest(genes)]
                expected=row['objectives'] if row['eligible'] else [-1e6,-1e6]
                if fit!=expected: raise ValueError('POPULATION_FITNESS_RECEIPT')
    eligible={k:v['objectives'] for k,v in receipts.rows.items() if v['eligible']}
    # Independent quadratic dominance check on unique objective vectors.
    keys=list(eligible);v=np.array([eligible[k] for k in keys]);unique=np.unique(v,axis=0)
    nondominated=set()
    for point in unique:
        if not np.any(np.all(unique>=point,axis=1)&np.any(unique>point,axis=1)):
            nondominated.add(tuple(point))
    independent={k for k,x in zip(keys,v) if tuple(x) in nondominated}
    if independent!=set(front['keys']): raise ValueError('INDEPENDENT_FRONTIER')
    order=sorted(independent,key=lambda k:(-eligible[k][0],-eligible[k][1],k))
    ranks=sorted({int(round(x)) for x in np.linspace(0,len(order)-1,min(5,len(order)))})
    if [order[i] for i in ranks]!=front['representatives']: raise ValueError('REPRESENTATIVE_RULE')
    engine(a.engine_root)
    if sources(a.engine_root)!=manifest['sources']: raise ValueError('SOURCE_PINS')
    cfg_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(cfg_path)!=manifest['config_sha256']: raise ValueError('RISK_CONFIG')
    cfg=json.loads(cfg_path.read_text())
    stored=json.loads(gzip.decompress((root/'REPLAYS.json.gz').read_bytes()))
    panels={name:[World(s,cfg) for s in panel_specs(seed)] for name,seed in [('calibration',7),('confirmation',43)]}
    genes={r['key']:r['genes'] for r in front['rows']}
    candidates=controls()+[(k,genes[k],None) for k in front['representatives']]
    if set(stored)!={c[0] for c in candidates}: raise ValueError('REPLAY_SET')
    count=0
    for name,g,enabled in candidates:
        for split,worlds in panels.items():
            result=evaluate(worlds,g,enabled,verify=True,details=True)
            if result!=stored[name][split]: raise ValueError(('NATIVE_REPLAY_MISMATCH',name,split))
            summary=next(r for r in report['summaries'] if r['name']==name and r['split']==split)
            if summary!=dict(name=name,split=split,**compact(result)): raise ValueError('SUMMARY_MISMATCH')
            count+=len(worlds)
    a.export.mkdir(parents=True)
    inventory={}
    for name in ['MANIFEST.json','REPORT.json','FRONTIER.json','REPLAYS.json.gz','SUMMARY.md']:
        shutil.copyfile(root/name,a.export/name);inventory[name]=dict(sha256=sha(a.export/name))
    for name in ['CANDIDATES.jsonl']+[f'SEED_{s}.json' for s in manifest['seeds']]:
        raw=(root/name).read_bytes();target=a.export/(name+'.gz')
        target.write_bytes(gzip.compress(raw,mtime=0))
        inventory[target.name]=dict(sha256=sha(target),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    verification=dict(status='PASS',identity=identity,receipt_count=len(receipts.rows),
        native_world_replays=count,exact_replay=True,full_budget_verified=True,
        frontier_policies=len(independent),unique_frontier_vectors=len(nondominated),
        representative_rule_verified=True,source_pins_verified=True,
        verifier_sha256=sha(Path(__file__)),artifacts=inventory)
    write(a.export/'VERIFICATION.json',verification)
    print(json.dumps({k:v for k,v in verification.items() if k!='artifacts'}),flush=True)


if __name__=='__main__': main()
