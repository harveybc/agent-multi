"""R4: independent re-verification of finalist matrices, in-memory tamper test, and re-simulation of sampled cells."""
import json,hashlib,sys,os,random,time
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D=sys.argv[1:5];NS=int(sys.argv[5])
sys.path.insert(0,SRC)
from pathlib import Path
import pathlib
from app.strategy_matrix_verification import verify
from app.rl_subset import digest
D=Path(D)
fin=json.loads((D/'FINALISTS_BEFORE_CONFIRMATION.json').read_text());ident=json.loads((D/'IDENTITY.json').read_text())
res=[]
for f in fin:
    key=digest(f['genes']);root=D/'finalists'/key/'matrix'
    t=time.time();v=verify(root);stored=json.loads((D/'finalists'/key/'VERIFICATION.json').read_text())
    mid=json.loads((root/'IDENTITY.json').read_text())
    strat=(D/'finalists'/key/'STRATEGY.json').read_bytes()
    r=dict(id=f['id'],verify_equal_stored=v==stored,genes_bound=mid['strategy_genes']==f['genes'],
        strategy_sha_bound=hashlib.sha256(strat).hexdigest()==mid['strategy_config_sha256'],
        agent_sources_equal_identity=mid['sources']['agent']==ident['sources'],
        engine_sources_equal_identity=mid['sources']['engine']==ident['engine_sources'],seconds=time.time()-t)
    # in-memory tamper: flip one stored chosen command in one cell
    cells=sorted((root/'cells').glob('*.json'));victim=cells[len(cells)//2]
    orig=pathlib.Path.read_text
    def patched(self,*a,**k):
        s=orig(self,*a,**k)
        if self==victim:
            j=json.loads(s);j['comparisons'][0]['chosen']=(j['comparisons'][0]['chosen']+1)%5;s=json.dumps(j)
        return s
    pathlib.Path.read_text=patched
    try:verify(root);r['tamper']='ACCEPTED'
    except ValueError as e:r['tamper']='REJECTED:'+str(e)
    finally:pathlib.Path.read_text=orig
    res.append(r);print(json.dumps(r),flush=True)
# re-simulate sampled cells
from tools.systematic_strategy_matrix import setup,run_cell
from app.directional_risk_calibration import DirectionalRisk
from app.recipe_runtime import attach_engine
recipe,config=setup(RECIPE,ENGINE)
rng=random.Random(20261027)
for f in fin:
    key=digest(f['genes']);root=D/'finalists'/key/'matrix';params=DirectionalRisk(tuple(f['genes']))
    mid=json.loads((root/'IDENTITY.json').read_text())
    allc=sorted((root/'cells').glob('*.json'));sample=rng.sample(allc,NS);eq=0;diffs=[]
    t=time.time()
    for p in sample:
        stored=json.loads(p.read_text());cell=stored['cell']
        new=dict(state='PASS',**run_cell(cell,config,recipe,None,parameters=params));new['identity_sha256']=digest(mid)
        new=json.loads(json.dumps(new,default=str))
        if new==stored:eq+=1
        else:diffs.append(p.stem)
    print(json.dumps(dict(id=f['id'],resimulated=NS,identical=eq,diffs=diffs[:5],seconds=time.time()-t)),flush=True)
