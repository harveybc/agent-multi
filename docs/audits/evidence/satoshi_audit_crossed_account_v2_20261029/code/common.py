"""Auditor's own helpers (independent of the producer's analyzer)."""
import gzip, json, os
from pathlib import Path
import numpy as np
S = Path(os.environ['S']); RUN = S/'run'
EVID = Path(os.environ['AM'])/'docs/audits/evidence/crossed_account_final_20261029'
QUALITIES = ['ideal', 'correlated', 'delayed', 'omitted_reversal', 'missing']
def unseal(p):
    return json.loads(Path(p).read_text())['data']
FRONT = unseal(RUN/'FRONTIER.json')
REPS = FRONT['representatives']
LABEL = {k: f'P{i+1}' for i, k in enumerate(REPS)}
for k in REPS: LABEL[k+'_size1'] = LABEL[k]+'_size1'
CONTROLS = ['nop', 'equal_guarded', 'greedy', 'only_3', 'only_6', 'only_12', 'only_24']
def names():
    out = list(CONTROLS)+list(REPS)
    out += [k+'_size1' for k in REPS if (RUN/f'REPLAY_correlated_{k}_size1.json.gz').exists()]
    return out
def label(n): return LABEL.get(n, n)
def replay(q, name):
    return json.loads(gzip.decompress((RUN/f'REPLAY_{q}_{name}.json.gz').read_bytes()))
def posthoc():
    return json.loads(gzip.decompress((EVID/'POST_HOC_P1_ONLY6.json.gz').read_bytes()))['result']
def cell_key(spec):
    return (spec['seed'], spec['pattern'], spec['volatility'], spec['spread_pips'], spec['mirror'])
def dump(name, obj):
    (S/'out'/name).write_text(json.dumps(obj, indent=1, sort_keys=True, default=float)+'\n')
