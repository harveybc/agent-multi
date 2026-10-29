"""Independent digest/receipt/population check (own code; imports nothing from agent-multi)."""
import hashlib, json, sys, gzip
from pathlib import Path
run, am, hs, ev = map(Path, sys.argv[1:5])
H=lambda b: hashlib.sha256(b).hexdigest()
def canon(v): return json.dumps(v, sort_keys=True, allow_nan=False).encode()
out={}
m=json.loads((run/'MANIFEST.json').read_text()); ident=m.pop('identity')
out['manifest_identity_recomputed']=H(canon(m)); out['manifest_identity_ok']=H(canon(m))==ident
mism=[]
for p,d in m['sources']['agent_multi'].items():
    if H((am/p).read_bytes())!=d: mism.append(('agent_multi',p))
hs_files={str(p.relative_to(hs)) for p in (hs/'app/short_horizon').rglob('*.py')}
if hs_files!=set(m['sources']['heuristic_strategy']): mism.append(('hs_fileset',sorted(hs_files^set(m['sources']['heuristic_strategy']))))
for p,d in m['sources']['heuristic_strategy'].items():
    if H((hs/p).read_bytes())!=d: mism.append(('heuristic_strategy',p))
out['source_counts']={k:len(v) for k,v in m['sources'].items()}; out['source_mismatches']=mism
out['config_sha_ok']=H((am/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_bytes())==m['config_sha256']
rep=json.loads((run/'REPORT.json').read_text())
for name,field in [('CANDIDATES.jsonl','receipts_sha256'),('REPLAYS.json.gz','replays_sha256'),('FRONTIER.json','frontier_sha256')]:
    out[f'{name}_digest_ok']=H((run/name).read_bytes())==rep[field]
ver=json.loads((ev/'VERIFICATION.json').read_text())
art={}
for n,v in ver['artifacts'].items():
    b=(ev/n).read_bytes(); ok=H(b)==v['sha256']
    if 'uncompressed_sha256' in v: ok=ok and H(gzip.decompress(b))==v['uncompressed_sha256']
    art[n]=ok
out['verification_artifacts_ok']=art
out['verifier_sha_ok']=H((am/'tools/verify_dynamic_account_results.py').read_bytes())==ver['verifier_sha256']
# receipt chain
chain=ident; rows={}; n=0; bad=0
for line in (run/'CANDIDATES.jsonl').read_text().splitlines():
    r=json.loads(line); seal=r.pop('receipt'); n+=1
    if r['previous']!=chain or r['identity']!=ident or H(canon(r))!=seal or r['key']!=H(canon(r['genes'])) or r['key'] in rows: bad+=1
    chain=seal; rows[r['key']]=r
out.update(receipts=n,receipt_bad=bad,receipt_tail_ok=chain==rep['receipt_tail'],unique_ok=len(rows)==rep['unique_candidates'])
# genes on decode grid
gridbad=0
B=m['bounds']
for r in rows.values():
    g=r['genes']
    if len(g)!=11 or any(not(lo-1e-12<=x<=hi+1e-12) for x,(lo,hi) in zip(g,B)) or any(abs(round(x,3)-x)>1e-12 for x in g[:10]) or g[10] not in (1,2,3): gridbad+=1
out['genes_off_grid_or_bounds']=gridbad
# population fitness
popbad=0; gens=0; seen=set()
for seed in m['seeds']:
    s=json.loads((run/f'SEED_{seed}.json').read_text())
    assert [h['generation'] for h in s['history']]==list(range(1,301))
    for h in s['history']:
        gens+=1
        for g,f in zip(h['genes'],h['fitness']):
            k=H(canon(g)); r=rows[k]['result']; seen.add(k)
            if f!=(r['objectives'] if r['eligible'] else [-1e6,-1e6]): popbad+=1
out.update(generations=gens,population_fitness_mismatch=popbad,keys_in_populations=len(seen))
out['eligible']=sum(r['result']['eligible'] for r in rows.values())
out['new_evaluations_sum']=sum(json.loads((run/f'SEED_{s}.json').read_text())['new_evaluations'] for s in m['seeds'])
print(json.dumps(out,indent=1))
