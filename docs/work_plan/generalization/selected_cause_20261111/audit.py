"""Independent NumPy audit of a sealed291 diagnostic and its290 TRAIN panel."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--diagnostic', type=Path, required=True)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.diagnostic
load = lambda p: json.loads(p.read_text())
receipt = load(root/'DONE.json')
for name, sha in receipt['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha
summary = load(root/'SUMMARY.json')
records = load(args.source/'panel/RECORDS.json')
data = np.load(root/'PROJECTIONS.npz')
z = data['logits']
p = np.exp(z-z.max(axis=1,keepdims=True)); p /= p.sum(axis=1,keepdims=True)
native = data['native_returns']; n = len(native)
best = native.argmax(axis=1)
checks = 0; max_error = 0.
for arm in summary['arms']:
    key = arm['arm']
    for name, control in arm['controls'].items():
        prefix = key+':'+name+':'
        pairs = [('total:batch_self','total:cross','total:total'),
                 ('adam:self','adam:cross','adam:adaptive'),
                 ('adam:adaptive','adam:decay','adam:total')]
        for a,b,c in pairs:
            err = np.max(np.abs(data[prefix+a]+data[prefix+b]-data[prefix+c]))
            assert err < 1e-12
            max_error = max(max_error,float(err)); checks += 1
        for group, stages in control['groups'].items():
            role = group.split(':')[0]
            mask = np.array([r['role']==role and (':' not in group or (r['position']==0 and r['allowed'])) for r in records])
            for stage, metrics in stages.items():
                delta = data[prefix+stage]
                full = np.column_stack([np.zeros(len(delta)),delta])
                dp = p*(full-(p*full).sum(axis=1,keepdims=True))
                values = dp[np.arange(n),best]
                assert int((mask & (values < -1e-10)).sum()) == metrics['lowers_native_optimal_probability']
                assert int((mask & (values > 1e-10)).sum()) == metrics['raises_native_optimal_probability']
                checks += 2
    for step in (1,8):
        prefix = f'{key}:endpoint{step}:'
        layer_sum = sum(data[k] for k in data.files if k.startswith(prefix+'layer:'))
        error = float(np.max(np.abs(layer_sum-data[prefix+'linear'])))
        assert error < 1e-12
        max_error = max(max_error,error); checks += 1
    for event in arm['endpoints']['8']['posthoc']:
        before = np.array(event['baseline_logits']); after = np.array(event['after_logits'])
        predicted = before[2]-before[0]+event['actual_margin_delta'][1]
        assert abs(after[2]-after[0]-predicted) < 1e-12
        checks += 1
assert summary['source_unchanged'] and not summary['new_candidate'] and not summary['test_read']
result = dict(identity=summary['identity'],receipt_files_verified=len(receipt['files']),
              independent_checks=checks,max_additive_error=max_error,
              source_unchanged=True,new_candidate=False,test_read=False)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
