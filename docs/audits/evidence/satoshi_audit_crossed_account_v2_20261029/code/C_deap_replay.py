"""Deterministic DEAP re-execution against the authenticated receipts (no candidate evaluation)."""
import json, os, sys, hashlib
from pathlib import Path
AM = Path(os.environ['AM']); sys.path.insert(0, str(AM))
S = Path(os.environ['S'])
from tools.optimize_dynamic_account import Receipts, search
run = S/'run'; man = json.loads((run/'MANIFEST.json').read_text()); ident = man['identity']
class Frozen(Receipts):
    def append(self, genes, result): raise RuntimeError('UNEXPECTED_NEW_EVALUATION')
rec = Frozen(run/'CANDIDATES.jsonl', ident)
od = S/'out'/'deap_replay'; od.mkdir(parents=True, exist_ok=True)
res = {}
for seed in man['seeds']:
    r = search([], rec, seed, 20, 300, od, ident, lambda: None, evaluate_fn=lambda *_: (_ for _ in ()).throw(RuntimeError('EVAL')))
    mine = json.loads((od/f'SEED_{seed}.json').read_text()); orig = json.loads((run/f'SEED_{seed}.json').read_text())
    res[seed] = dict(history_equal=mine['history'] == orig['history'], new_evaluations=r['new_evaluations'],
                     orig_new_evaluations=orig['new_evaluations'],
                     bytes_equal=(od/f'SEED_{seed}.json').read_bytes() == (run/f'SEED_{seed}.json').read_bytes())
    print(seed, res[seed], flush=True)
(S/'out'/'C_deap_replay.json').write_text(json.dumps(res, indent=1)+'\n')
