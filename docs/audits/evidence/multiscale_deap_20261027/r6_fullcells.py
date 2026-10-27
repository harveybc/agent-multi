"""R6: re-simulate a chunk of ALL matrix cells of one finalist and compare to the stored receipts."""
import json,sys,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D,FID,K,N=sys.argv[1:8];K=int(K);N=int(N)
sys.path.insert(0,SRC)
from pathlib import Path
from app.rl_subset import digest
from tools.systematic_strategy_matrix import setup,run_cell
from app.directional_risk_calibration import DirectionalRisk
D=Path(D);fin={f['id']:f for f in json.loads((D/'FINALISTS_BEFORE_CONFIRMATION.json').read_text())}[FID]
key=digest(fin['genes']);root=D/'finalists'/key/'matrix';mid=json.loads((root/'IDENTITY.json').read_text());IDM=digest(mid)
recipe,config=setup(RECIPE,ENGINE);params=DirectionalRisk(tuple(fin['genes']))
allc=sorted((root/'cells').glob('*.json'))[K::N];eq=0;diff=[];t=time.time()
for p in allc:
    stored=json.loads(p.read_text())
    new=dict(state='PASS',**run_cell(stored['cell'],config,recipe,None,parameters=params));new['identity_sha256']=IDM
    if json.loads(json.dumps(new,default=str))==stored:eq+=1
    else:diff.append(p.stem)
print(json.dumps(dict(id=FID,chunk=K,of=N,cells=len(allc),identical=eq,diffs=diff[:10],seconds=time.time()-t)))
