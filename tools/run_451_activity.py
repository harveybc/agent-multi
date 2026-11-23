"""Paired continuation of the active validation champion, semesters 6 through 8."""
import fcntl
import shutil
from pathlib import Path

from run_448_campaign import HOST, ROOT, atomic, base, launch, run

OUT=Path('/var/tmp/role451-activity')
CONTROL=Path('/var/tmp/role450-full-policy/run-7')


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    dest=OUT/'run-7'
    dest.mkdir(exist_ok=True)
    worker=ROOT/'tools/role448_worker.py'
    parent=CONTROL/'s4-validation.pt'
    spec=dict(seed=7,arm='validation',stages=[6,7,8],parent_sha=base.file_sha(parent),
              worker_sha=base.file_sha(worker),eligibility='trades > 2 * validation calendar weeks',
              score='validation return; strict increase > 1e-8',promoted=False)
    if (OUT/'PROTOCOL.json').exists():assert base.read(OUT/'PROTOCOL.json')==spec
    atomic(OUT/'PROTOCOL.json',spec)
    shutil.copyfile(worker,OUT/worker.name)
    shutil.copyfile(CONTROL/'s5-validation-entry.json',OUT/'control-entry.json')
    for p in (parent,parent.with_suffix('.json')):shutil.copyfile(p,dest/p.name)
    run(['ssh',HOST,f'mkdir -p {dest}'])
    run(['rsync','-a',str(dest/parent.name),str(dest/parent.with_suffix('.json').name),f'{HOST}:{dest}/'])
    run(['rsync','-a',str(OUT/worker.name),str(OUT/'control-entry.json'),str(OUT/'PROTOCOL.json'),f'{HOST}:{OUT}/'])
    launch(7,1,True,OUT,True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    stages=[]
    for s in (5,6,7):
        name=f's{s}-validation'
        result=base.read(dest/f'{name}.json')
        control=base.read(CONTROL/f'{name}.json')
        hist=torch.load(dest/f'{name}.pt',map_location='cpu',weights_only=False)['history']
        fig,(a,b)=plt.subplots(2,1,sharex=True)
        a.plot([r['val']['ret'] for r in hist],label='Validation')
        for j in range(s):a.plot([r['prior'][j]['ret'] for r in hist],alpha=.5,label=f'Train {j+1}')
        a.set_ylabel('Return (%)');a.legend(ncol=3)
        b.plot([r['val']['orders_week'] for r in hist])
        b.axhline(2,linestyle=':');b.set(xlabel='Epoch',ylabel='Validation sells/week')
        for ax in (a,b):ax.axvline(result['best'],color='black',linestyle=':')
        fig.tight_layout();fig.savefig(dest/f'{name}.png');plt.close(fig)
        assert result['val']['orders_week']>2
        stages.append(dict(semester=s+1,control=control['val'],trial=result,rollbacks=sum(r.get('rollback',False) for r in hist)))
    atomic(OUT/'DONE.json',dict(stages=stages,improved_over_parent=stages[-1]['trial']['metric']>
          base.read(CONTROL/'s4-validation.json')['metric']+1e-8,promoted=False))


if __name__=='__main__':main()
