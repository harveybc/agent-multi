"""Own frontier: Kung-style divide&conquer on maxima + explicit tie/duplicate semantics (no agent-multi import)."""
import json, sys, hashlib, math
from pathlib import Path
from collections import Counter, defaultdict
run=Path(sys.argv[1])
rows={}
for l in (run/'CANDIDATES.jsonl').read_text().splitlines():
    r=json.loads(l); rows[r['key']]=r
def maxima(points):
    """Kung: return set of vectors (x,y) not weakly-dominated-with-strict-improvement by any other."""
    pts=sorted(set(points))  # ascending x then y
    if len(pts)<=1: return set(pts)
    mid=len(pts)//2
    # split by x value boundary so equal x stay together
    xs=[p[0] for p in pts]; cut=mid
    while cut<len(pts) and xs[cut]==xs[cut-1]: cut+=1
    if cut==len(pts):
        cut=mid
        while cut>0 and xs[cut]==xs[cut-1]: cut-=1
    if cut==0:  # all same x: max y only
        y=max(p[1] for p in pts); return {p for p in pts if p[1]==y}
    L=maxima(pts[:cut]); R=maxima(pts[cut:])
    ymaxR=max(p[1] for p in R)
    # left points (smaller x) survive only if y strictly greater than every right maximum
    return R|{p for p in L if p[1]>ymaxR}
elig={k:tuple(r['result']['objectives']) for k,r in rows.items() if r['result']['eligible']}
allv={k:tuple(r['result']['objectives']) for k,r in rows.items()}
F=maxima(list(elig.values()))
# brute check on distinct vectors
U=set(elig.values())
brute={p for p in U if not any(q[0]>=p[0] and q[1]>=p[1] and q!=p for q in U)}
assert brute==F
keys=sorted(k for k,v in elig.items() if v in F)
front=json.loads((run/'FRONTIER.json').read_text())
out=dict(eligible=len(elig),ineligible=len(rows)-len(elig),frontier_policies=len(keys),distinct_vectors=len(F),
         same_keyset=set(keys)==set(front['keys']),same_order=None)
mult=Counter(elig[k] for k in keys)
out['duplicate_vector_multiplicity']=dict(Counter(mult.values()))
# ineligible that would dominate
Fall=maxima(list(allv.values()))
inel_dom=[k for k,v in allv.items() if not rows[k]['result']['eligible'] and v in Fall]
out['ineligible_on_unconstrained_frontier']=len(inel_dom)
out['ineligible_examples']=[dict(key=k[:12],obj=allv[k],mean_orders=rows[k]['result']['mean_orders']) for k in inel_dom[:5]]
# representative rule: rank by (-ret,-negdd,key), 5 evenly spaced positions
order=sorted(keys,key=lambda k:(-elig[k][0],-elig[k][1],k))
out['same_order']=order==front['keys']
n=len(order); pos=sorted({int(math.floor(i*(n-1)/4+0.5)) for i in range(5)})
reps=[order[i] for i in pos]
out['rep_positions']=pos; out['reps_match']=reps==front['representatives']
vec_rank={v:i for i,v in enumerate(sorted(F,key=lambda v:(-v[0],-v[1])))}
out['reps']=[dict(alias=f'P{i+1}',key=k[:12],genes=rows[k]['genes'],obj=elig[k],distinct_vector_rank=vec_rank[elig[k]],multiplicity=mult[elig[k]],mean_orders=rows[k]['result']['mean_orders']) for i,k in enumerate(reps)]
# numpy round vs half-even check
import numpy as np
out['numpy_linspace_round_positions']=sorted({int(round(x)) for x in np.linspace(0,n-1,5)})
# tie semantics: pairs with equal return different dd, and equal dd different return on frontier
xs=Counter(v[0] for v in F); ys=Counter(v[1] for v in F)
out['frontier_equal_x']=sum(c>1 for c in xs.values()); out['frontier_equal_y']=sum(c>1 for c in ys.values())
# gene-vector duplicates mapping to same objective: how many distinct size genes among duplicates
groups=defaultdict(list)
for k in keys: groups[elig[k]].append(rows[k]['genes'])
big=max(groups.items(),key=lambda kv:len(kv[1]))
out['largest_duplicate_group']=dict(vector=big[0],policies=len(big[1]),distinct_gene_values_per_gene=[len({g[i] for g in big[1]}) for i in range(11)])
print(json.dumps(out,indent=1))
