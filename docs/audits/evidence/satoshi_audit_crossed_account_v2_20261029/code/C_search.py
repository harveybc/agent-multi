"""C: search reproducibility: receipt chain, eligibility, brute-force nondominance, rep rule, DEAP replay."""
import hashlib, json, sys, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
def dg(v): return hashlib.sha256(json.dumps(v, sort_keys=True, allow_nan=False).encode()).hexdigest()
man = json.loads((RUN/'MANIFEST.json').read_text()); ident = man['identity']
body = {k: v for k, v in man.items() if k != 'identity'}
out = dict(manifest_identity_ok=dg(body) == ident)
rows = {}; chain = ident; bad = []
for i, line in enumerate((RUN/'CANDIDATES.jsonl').read_text().splitlines()):
    r = json.loads(line); seal = r.pop('receipt')
    if r['previous'] != chain or dg(r) != seal or r['key'] != dg(r['genes']) or r['key'] in rows: bad.append(i)
    chain = seal; rows[r['key']] = r
out.update(receipts=len(rows), chain_breaks=len(bad), tail=chain)
comp = unseal(RUN/'COMPLETE.json'); out['tail_matches_complete'] = chain == comp['receipt_tail']
# eligibility
elig_bad = [k for k, r in rows.items() if r['result']['eligible'] != (r['result']['mean_orders'] >= 3)
            or abs(r['result']['orders']/360-r['result']['mean_orders']) > 1e-12]
out['eligibility_inconsistent'] = len(elig_bad)
E = [k for k, r in rows.items() if r['result']['eligible']]
out['eligible'] = len(E); out['ineligible'] = len(rows)-len(E)
# genes decoded? check bounds/rounding
B = man['bounds']
gbad = [k for k, r in rows.items() if any(not (lo-1e-12 <= g <= hi+1e-12) or abs(round(g, 3)-g) > 1e-12 for g, (lo, hi) in zip(r['genes'], B)) or not isinstance(r['genes'][-1], int)]
out['genes_out_of_bounds_or_unrounded'] = len(gbad)
V = np.array([rows[k]['result']['objectives'] for k in E])
# brute force: j dominates i iff V[j]>=V[i] all and > any
dominated = np.zeros(len(E), bool)
for s in range(0, len(E), 500):
    vi = V[s:s+500][:, None, :]
    ge = (V[None] >= vi).all(-1); gt = (V[None] > vi).any(-1)
    dominated[s:s+500] = (ge & gt).any(1)
nd = sorted(E[i] for i in np.flatnonzero(~dominated))
out['nondominated_bruteforce'] = len(nd)
out['matches_FRONTIER_keys'] = set(nd) == set(FRONT['keys'])
groups = defaultdict(list)
for k in nd: groups[tuple(rows[k]['result']['objectives'])].append(k)
out['distinct_vectors'] = len(groups)
out['duplicate_groups'] = {str(n): sum(len(v) == n for v in groups.values()) for n in sorted({len(v) for v in groups.values()})}
ordered = sorted(groups, key=lambda v: (-v[0], -v[1]))
pos = np.linspace(0, len(ordered)-1, 5)
# Python round() is half-to-even; report both
ranks_even = sorted({round(x) for x in pos}); ranks_up = sorted({int(np.floor(x+.5)) for x in pos})
out['rank_positions'] = pos.tolist(); out['ranks_half_even'] = ranks_even; out['ranks_half_up'] = ranks_up
reps = [min(groups[ordered[i]], key=lambda k: tuple(rows[k]['genes'])) for i in ranks_even]
out['representatives_match'] = reps == REPS
out['rep_groups'] = [dict(label=f'P{i+1}', vector=list(ordered[r]), group_size=len(groups[ordered[r]]),
                          members=[rows[k]['genes'] for k in groups[ordered[r]]]) for i, r in enumerate(ranks_even)]
# frontier gene stats
G = np.array([rows[k]['genes'] for k in nd], float)
out['frontier_gene_stats'] = {n: dict(min=float(G[:, j].min()), max=float(G[:, j].max()), median=float(np.median(G[:, j])))
                              for j, n in enumerate(man['genes'])}
out['frontier_size_gene_eq_1'] = int((G[:, 8] == 1).sum()); out['frontier_persistence_eq_1'] = int((G[:, 10] == 1).sum())
x = np.array([rows[k]['result']['objectives'][0] for k in nd])
out['corr_size_gene_vs_return_frontier'] = float(np.corrcoef(G[:, 8], x)[0, 1])
# history fitness
hist_bad = 0; n = 0
for seed in man['seeds']:
    h = json.loads((RUN/f'SEED_{seed}.json').read_text())['history']
    for step in h:
        for g, f in zip(step['genes'], step['fitness']):
            r = rows[dg(g)]['result']; n += 1
            if f != (r['objectives'] if r['eligible'] else [-1e6, -1e6]): hist_bad += 1
out.update(history_entries=n, history_fitness_mismatch=hist_bad)
# where were representatives found
first = {}
for seed in man['seeds']:
    h = json.loads((RUN/f'SEED_{seed}.json').read_text())['history']
    for step in h:
        for g in step['genes']:
            first.setdefault(dg(g), (seed, step['generation']))
out['rep_first_in_population'] = {f'P{i+1}': first.get(k) for i, k in enumerate(REPS)}
out['frontier_by_seed_firstseen'] = {str(s): sum(1 for k in nd if first.get(k, (None,))[0] == s) for s in man['seeds']}
out['frontier_never_in_saved_population'] = sum(1 for k in nd if k not in first)
dump('C_search.json', out)
print(json.dumps({k: v for k, v in out.items() if k not in ('rep_groups', 'frontier_gene_stats')}, default=str))
