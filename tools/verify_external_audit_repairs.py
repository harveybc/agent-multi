#!/usr/bin/env python3
"""Replay the immutable predecessor under audit repairs, not a new optimization."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.dynamic_account_panel import World, evaluate
from app.zigzag_opportunities import engine, sha
from tools.optimize_dynamic_account import digest, sources
from tools.verify_absolute_early_close import write

PREDECESSOR='098e871909f0c587b1cbe855cc49b93723ed63a03c74845a52dd9009071a966d'
CHANGED={'app/protection_reference.py','app/close_hold_controller.py','app/dynamic_account_panel.py'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--test-report',required=True,type=Path,help='Focused pytest JUnit XML; only safe test fields are exported')
    a=p.parse_args()
    if a.output.exists(): raise ValueError('NEW_OUTPUT_REQUIRED')
    cases=ET.parse(a.test_report).getroot().findall('.//testcase')
    if not cases or any(c.find(tag) is not None for c in cases for tag in ('failure','error','skipped')):
        raise ValueError('FOCUSED_TESTS_NOT_ALL_PASSED')
    test_files=sorted({c.attrib['classname'].replace('.','/')+'.py' for c in cases})
    tests=dict(status='PASS',count=len(cases),
        nodes=[c.attrib['classname'].replace('.','/')+'.py::'+c.attrib['name'] for c in cases],
        sources={f:sha(ROOT/f) for f in test_files})
    root=ROOT/'docs/audits/evidence/dynamic_account_pareto_20261029'
    manifest=json.loads((root/'MANIFEST.json').read_text())
    identity=manifest.pop('identity')
    if identity!=PREDECESSOR or digest(manifest)!=identity: raise ValueError('PREDECESSOR_IDENTITY')
    report=json.loads((root/'REPORT.json').read_text())
    for name,field in [('FRONTIER.json','frontier_sha256'),('REPLAYS.json.gz','replays_sha256')]:
        if sha(root/name)!=report[field]: raise ValueError('ARTIFACT_DIGEST')
    engine(a.engine_root)
    current=sources(a.engine_root)
    if current['heuristic_strategy']!=manifest['sources']['heuristic_strategy']:
        raise ValueError('NATIVE_ENGINE_CHANGED')
    changed={k for k,v in current['agent_multi'].items() if v!=manifest['sources']['agent_multi'][k]}
    if changed!=CHANGED: raise ValueError(('UNEXPECTED_REPAIR_SCOPE',sorted(changed)))
    config=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(config)!=manifest['config_sha256']: raise ValueError('CONFIG_CHANGED')
    cfg=json.loads(config.read_text());front=json.loads((root/'FRONTIER.json').read_text())
    old=json.loads(gzip.decompress((root/'REPLAYS.json.gz').read_bytes()))
    genes={r['key']:r['genes'] for r in front['rows']}
    candidates=[(c['name'],c['genes'],{tuple(x) for x in c['enabled']}) for c in manifest['controls']]
    candidates.extend((k,genes[k],None) for k in front['representatives'])
    if set(old)!={x[0] for x in candidates}: raise ValueError('CANDIDATE_SET')
    panels={s:[World(spec,cfg) for spec in manifest[s]] for s in ('calibration','confirmation')}
    count=0
    for name,g,enabled in candidates:
        for split,worlds in panels.items():
            result=evaluate(worlds,g,enabled,verify=True,details=True)
            if result!=old[name][split]: raise ValueError(('REPAIRED_REPLAY_DIFFERS',name,split))
            count+=len(worlds)
        print(json.dumps(dict(policy=name,accounts_verified=count)),flush=True)
    p1=front['representatives'][0]
    paired=[]
    for candidate,control in zip(old[p1]['confirmation']['worlds'],old['only_3']['confirmation']['worlds']):
        if candidate['spec']!=control['spec']: raise ValueError('UNPAIRED_SUPPORT')
        paired.append(dict(spec=candidate['spec'],p1_return=candidate['return_pct'],
            only3_return=control['return_pct'],delta_pp=candidate['return_pct']-control['return_pct']))
    evidence=dict(status='PASS',predecessor=identity,accounts_replayed=count,
        exact_full_account_equality=True,new_search=False,learned_reporter=False,
        source_changes=sorted(changed),sources=current,verifier_sha256=sha(Path(__file__)),
        focused_tests=tests,
        predecessor_artifacts={n:sha(root/n) for n in ('MANIFEST.json','REPORT.json','FRONTIER.json','REPLAYS.json.gz')},
        paired_p1_only3_confirmation=paired)
    a.output.mkdir(parents=True)
    write(a.output/'REPAIR_VERIFICATION.json',evidence)
    print(json.dumps(dict(status='PASS',accounts=count)),flush=True)


if __name__=='__main__': main()
