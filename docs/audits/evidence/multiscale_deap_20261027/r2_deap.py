"""R2: re-run the nine DEAP searches from scratch (cache-backed evaluator; misses simulated) and a split resume."""
import json,hashlib,sys,os,shutil
os.environ['CUDA_VISIBLE_DEVICES']=''
SRC,ENGINE,RECIPE,D,TMP=sys.argv[1:6]
sys.path.insert(0,SRC)
from pathlib import Path
import app.multiscale_strategy_search as M
from tools.deap_output_calibration import evolve
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
D=Path(D);TMP=Path(TMP);identity=json.loads((D/'IDENTITY.json').read_text())
cfg=identity['search'];scales=json.loads((D/'SCALES.json').read_text())
M.initialize(RECIPE,ENGINE)
misses=[0];hits=[0]
def report(genes):
    k=digest(dict(genes=genes,split='calibration',mode='ideal'));p=D/'cache'/(k+'.json')
    if p.exists():hits[0]+=1;return json.loads(p.read_text())['payload']['report']
    misses[0]+=1
    return M.aggregate([M.episode((c,genes,'ideal')) for c in M.panel('calibration')])
search_cfg={k:cfg[k] for k in ('population','generations','patience','tolerance')};search_cfg['initial_controls']=cfg['controls']
out=[]
for seed in cfg['seeds']:
  for name in cfg['objectives']:
    cur=f'{name}_s{seed}'
    def ev(genes,name=name):
        r=report(genes);return M.score(name,r['components'],scales) if r['eligible'] else -1e9-len(r['reasons'])
    folder=TMP/cur
    if folder.exists():shutil.rmtree(folder)
    st=evolve(folder,seed,ev,search_cfg,identity,objective_name=name)
    sealed=json.loads((D/'searches'/cur/'STATE.json').read_text())
    mine=json.loads((folder/'STATE.json').read_text())
    out.append(dict(id=cur,state_file_equal=(folder/'STATE.json').read_bytes()==(D/'searches'/cur/'STATE.json').read_bytes(),
        payload_digest_equal=mine['payload_sha256']==sealed['payload_sha256'],best=st['best']==sealed['payload']['best'],
        generation=st['generation'],stop=st['stop_reason']))
    print(json.dumps(out[-1]),flush=True)
# split resume for sharpe_s7 and sharpe_return_activity_s43
for seed,name in ((7,'sharpe'),(43,'sharpe_return_activity')):
    cur=f'{name}_s{seed}';folder=TMP/('resume_'+cur)
    if folder.exists():shutil.rmtree(folder)
    def ev(genes,name=name):
        r=report(genes);return M.score(name,r['components'],scales) if r['eligible'] else -1e9-len(r['reasons'])
    evolve(folder,seed,ev,search_cfg,identity,stop_after=37,objective_name=name)
    mid=json.loads((folder/'STATE.json').read_text())['payload']['generation']
    evolve(folder,seed,ev,search_cfg,identity,objective_name=name)
    sealed=json.loads((D/'searches'/cur/'STATE.json').read_text())
    mine=json.loads((folder/'STATE.json').read_text())
    print(json.dumps(dict(resume=cur,stopped_at=mid,payload_digest_equal=mine['payload_sha256']==sealed['payload_sha256'])),flush=True)
print(json.dumps(dict(cache_hits=hits[0],cache_misses=misses[0])))
