"""A3: index every sealed cache entry (genes, split, mode, report summary), verify seals."""
import json,hashlib,sys
from pathlib import Path
D=Path(sys.argv[1])
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
IDH=digest(json.loads((D/'IDENTITY.json').read_text()))
bad=0;rows=[]
for p in sorted((D/'cache').glob('*.json')):
    r=json.loads(p.read_text());pl=r['payload']
    ok=r['identity_sha256']==IDH and r['payload_sha256']==digest(pl) and p.stem==digest(dict(genes=pl['genes'],split=pl['split'],mode=pl['mode']))
    bad+=not ok
    tapes=[e['tape_sha256'] for e in pl['episodes']]
    rep=pl['report']
    rows.append(dict(key=p.stem,genes=pl['genes'],split=pl['split'],mode=pl['mode'],ok=ok,
        comp=rep['components'],eligible=rep['eligible'],reasons=rep['reasons'],trades=rep['trades'],
        close=rep['close_reasons'],sides={k:v['trades'] for k,v in rep['sides'].items()},tapes=digest(tapes)))
json.dump(dict(bad=bad,n=len(rows),rows=rows),open(sys.argv[2],'w'))
print(bad,len(rows))
