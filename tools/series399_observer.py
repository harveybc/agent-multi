"""Relay compact SAC receipts and distribute completed-generator diagnostics."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import statistics
import time


ROOT='/var/tmp/sac399-series'
AUD=Path(ROOT)/'audit'
RECEIPTS=Path(ROOT)/'receipts'
SRC='/var/tmp/prefix398-dragon-61'
HOSTS=['','gamma-lan','dragon-lan']
PY='/var/tmp/discrete-sac-probe-venv/bin/python'


def run(args,host=''):
    return subprocess.check_output((['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',host] if host else [])+
                                   list(map(str,args)),text=True,timeout=180)


def write(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    tmp.replace(path)


def snapshot(host,arm):
    target=RECEIPTS/arm
    target.mkdir(exist_ok=True,parents=True)
    run(['rsync','-a','--include=/*.json','--exclude=*',(host+':' if host else '')+ROOT+'/runs/'+arm+'/',target])
    return json.loads((target/'PROGRESS.json').read_text()) if (target/'PROGRESS.json').exists() else {}


def main():
    arms=[('gamma-lan','8y-s7'),('dragon-lan','8y-s19'),('gamma-lan','4y-s7'),('','4y-s19')]
    progress={a:snapshot(h,a) for h,a in arms}
    status={}
    for h,a in arms:
        health=run(['systemctl','--user','show','sac399-'+a+'.service','-p','ActiveState','-p','Result'],h)
        r=progress[a]
        status[a]=dict({k:r.get(k) for k in ('stage','epoch','steps','updates','decisions_this_epoch','epoch_seconds')},
                       service=dict(x.split('=',1) for x in health.splitlines() if '=' in x),
                       completed=(RECEIPTS/a/'RESULT.json').exists())
    write(RECEIPTS/'STATUS.json',dict(at=time.time(),arms=status))
    snapshots=[]
    for _,a in arms:
        history=RECEIPTS/a/'HISTORY.json'
        if history.is_file():
            for r in json.loads(history.read_text()):
                snapshots.append(dict(arm=a,epoch=r['epoch'],updates=r['updates'],steps=r['steps'],
                    train=r['metrics']['TRAIN']['return_pct'],validation=r['metrics']['VALIDATION']['return_pct'],
                    val_trades=r['metrics']['VALIDATION']['trades'],val_sharpe=r['metrics']['VALIDATION']['sharpe'],
                    patience_used=r['monitor']['bad'],epoch_seconds=r.get('epoch_seconds')))
    write(RECEIPTS/'EPOCHS.json',snapshots)
    cut=json.loads(run(['cat',SRC+'/PROGRESS.json'],'dragon-lan'))
    limit=cut['rows']
    report=AUD/f'SNAPSHOT-{limit}.json'
    if report.exists(): return
    run(['rsync','-a','--include=/BINDING.json','--include=/part-*/','--include=*.json',
         '--include=*.parquet','--exclude=*','dragon-lan:'+SRC+'/',str(AUD/'parts')+'/'])
    run(['rsync','-a',str(AUD/'parts')+'/', 'gamma-lan:'+ROOT+'/audit/parts/'])
    def one(j):
        src=SRC if j==2 else str(AUD/'parts')
        result=json.loads(run(['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',PY,
                       ROOT+'/runtime/tools/prefix399_audit.py',src,AUD/'shared',limit,j],HOSTS[j]))
        return result
    with ThreadPoolExecutor(3) as pool: shards=list(pool.map(one,range(3)))
    legs=[leg for s in shards for leg in s['legs']]
    if sum(s['rows'] for s in shards)!=limit or len({r[0] for r in legs})!=len(legs): raise ValueError('shard coverage')
    detail=dict(generated=cut,shards=[{k:v for k,v in s.items() if k!='legs'} for s in shards],
        complete_legs=len(legs),endpoint_mae_bps=statistics.mean(r[2] for r in legs),
        extreme_at_last_fraction=statistics.mean(r[3]==0 for r in legs),first_seed_leg_excluded=True,
        sac_release=False)
    write(report,detail)
    write(AUD/'LATEST.json',detail)


if __name__=='__main__': main()
