"""Regression: recompute all 8,720 close/hold rollouts (plan 133 harness) with PATCHED code and compare record-by-record
with the committed CASES.json.gz (produced by pre-repair code). Also the 25 prefix-invariance witnesses.
Usage: python -I D_closehold_8720.py <am> <hs>"""
import gzip, json, sys, time
from pathlib import Path
AM, HS = Path(sys.argv[1]), Path(sys.argv[2]); sys.path.insert(0, str(AM))
from app.zigzag_opportunities import engine; engine(HS)
from app.absolute_exit_corpus import specs
from app.close_hold_verification import arms, run_case, prefix_future_witness
root = AM/'docs/audits/evidence/absolute_early_close_20261029'
refs = json.loads(gzip.decompress((root/'CASES.json.gz').read_bytes()))
config = json.loads((AM/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
old = json.loads(gzip.decompress((AM/'docs/audits/evidence/close_hold_controller_20261029/CASES.json.gz').read_bytes()))
t0 = time.monotonic(); records = []; witnesses = []
for i, r in enumerate(refs):
    if i % 72 == 0: witnesses.append(prefix_future_witness(r['spec'], config))
    for arm in arms(): records.append(run_case(r['spec'], config, r, arm))
new = json.loads(json.dumps(records, sort_keys=True, allow_nan=False))
diff = [i for i, (a, b) in enumerate(zip(new, old)) if a != b]
committed_report = json.loads((AM/'docs/audits/evidence/close_hold_controller_20261029/REPORT.json').read_text())
print(json.dumps(dict(records=len(new), committed=len(old), mismatching_records=len(diff), first=diff[:5],
    witnesses=len(witnesses), witnesses_equal=witnesses == committed_report['prefix_invariance_witnesses'],
    bytes_equal=(json.dumps(records, sort_keys=True, allow_nan=False)+'\n').encode() ==
                gzip.decompress((AM/'docs/audits/evidence/close_hold_controller_20261029/CASES.json.gz').read_bytes()),
    seconds=time.monotonic()-t0), indent=1))

# --- diagnosis of mismatches (appended after the first run showed 2,450 differing records)
from collections import Counter
def walk(a, b, path=''):
    if type(a) != type(b): yield path, a, b; return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k): yield from walk(a.get(k), b.get(k), f'{path}.{k}')
    elif isinstance(a, list):
        if len(a) != len(b): yield path+'[len]', len(a), len(b); return
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y: yield from walk(x, y, f'{path}[]')
    else: yield path, a, b
fields = Counter(); arms_c = Counter(); modes = Counter(); maxabs = {}; actions_changed = 0; exits_changed = 0
for i in diff:
    a, b = new[i], old[i]; arms_c[a['arm']] += 1; modes[(a['spec']['mode'], a['spec']['horizon'])] += 1
    if [t.get('action') for t in a.get('trace', [])] != [t.get('action') for t in b.get('trace', [])]: actions_changed += 1
    if a.get('metrics') != b.get('metrics'): exits_changed += 1
    for p, x, y in walk(a, b):
        fields[p] += 1
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            maxabs[p] = max(maxabs.get(p, 0.), abs(x-y))
print(json.dumps(dict(arms=arms_c, modes={str(k): v for k, v in modes.items()}, fields=fields, maxabs=maxabs,
                      records_with_action_sequence_changed=actions_changed, records_with_metrics_changed=exits_changed,
                      example_keys=sorted(new[diff[0]].keys())), indent=1, default=str))
