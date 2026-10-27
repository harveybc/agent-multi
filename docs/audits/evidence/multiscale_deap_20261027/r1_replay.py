"""R1: re-simulate genomes with the pinned engine and compare bitwise to sealed cache ledgers."""
import json,hashlib,sys,time,os
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D=sys.argv[1:5];which=sys.argv[5].split(',');splits=sys.argv[6].split(',')
sys.path.insert(0,SRC)
from pathlib import Path
import app.multiscale_strategy_search as M
from tools.recipe_sac_queue import clean
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
D=Path(D);identity=json.loads((D/'IDENTITY.json').read_text());IDH=digest(identity)
M.initialize(RECIPE,ENGINE)
arms={r['id']:r['genes'] for r in json.loads((D/'SEARCH_RESULTS.json').read_text())}
arms.update({r['id']:r['genes'] for r in json.loads((D/'CONTROLS.json').read_text())})
res=[]
for aid in which:
  genes=arms[aid]
  for spec in splits:
    split,mode=(spec.split(':')+['ideal'])[:2]
    t=time.time()
    rows=[M.episode((cell,genes,mode)) for cell in M.panel(split)]
    rep=M.aggregate(rows)
    k=digest(dict(genes=genes,split=split,mode=mode));sealed=json.loads((D/'cache'/(k+'.json')).read_text())['payload']
    new=json.loads(json.dumps(dict(genes=genes,split=split,mode=mode,report=rep,episodes=rows),default=str))
    same_payload=digest(new)==digest(sealed)
    same_trades=all(json.dumps(a['trades'],sort_keys=True,default=str)==json.dumps(b['trades'],sort_keys=True,default=str) for a,b in zip(new['episodes'],sealed['episodes']))
    same_report=json.dumps(new['report'],sort_keys=True)==json.dumps(sealed['report'],sort_keys=True)
    res.append(dict(arm=aid,split=split,mode=mode,payload_digest_equal=same_payload,trades_equal=same_trades,report_equal=same_report,
        sharpe=rep['components']['sharpe'],seconds=time.time()-t))
    print(json.dumps(res[-1]),flush=True)
