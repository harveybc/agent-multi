"""Export small public receipts and training curves; exclude raw market data and tensors."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd


def report(root, out, host):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    read=lambda p:json.loads(p.read_text())
    out.mkdir(exist_ok=True,parents=True)
    accounts=[]; scores=[]; details=[]; curves=[]
    chart,axes=plt.subplots(4,4,figsize=(15,10),constrained_layout=True)
    for year in range(2020,2024):
        r=root/str(year)
        a=read(r/'audit.json'); ready=read(r/'ready.json')
        details.append({k:a[k] for k in ('year','start','monitor','last_fit_close','eval_start','last_eval_close',
            'fit_bars','windows','prefix_checks','parity_checked_rows','parity_failures')}|ready)
        unit={2020:'role477-5090',2021:'role477-2021',2022:'role477-2022',2023:'role477-5090'}[year]
        cmd=['journalctl','--user','-u',unit,'--no-pager','-o','cat']
        lines=subprocess.check_output(cmd if year==2022 else ['ssh',host,*cmd],text=True).splitlines()
        events=[json.loads(line) for line in lines if line.startswith('{"epoch"')]
        for task,j in [('short',0),('long',2)]:
            done=read(r/task/'done.json')
            status=read(r/task/'status.json')
            history={e['epoch']:e for e in events if e['train_rows']==status['train_rows']
                     and e['monitor_rows']==status['monitor_rows']}
            assert set(history)==set(range(1,done['epoch']+1))
            h=pd.DataFrame([history[n] for n in sorted(history)])
            assert abs(history[done['best_epoch']]['monitor_mae']-status['best'])<1e-12
            assert 0 <= status['best']-h.monitor_mae.min() <= 1e-8
            curves.append(h.assign(year=year,task=task))
            for col,k in [('train_scaled_l1',0),('monitor_mae',1)]:
                ax=axes[year-2020,j+k];ax.plot(h.epoch,h[col],lw=1)
                ax.axvline(done['best_epoch'],color='green',ls=':',lw=1)
                for ep in h.epoch[h.rollback]:ax.axvline(ep,color='grey',ls=':',lw=.7)
                ax.set_title(f'{year} {task} {col}',fontsize=9);ax.set_xlabel('epoch')
            for h,v,b in zip(range(1,7) if task=='short' else (24,48),done['mae'],
                             done['persistence_mae'],strict=True):
                scores.append(dict(year=year,task=task,h=h,rows=done['holdout_rows'],model_mae=v,
                    persistence_mae=b,skill_pct=100*(1-v/b),epochs=done['epoch'],champ=done['best_epoch'],
                    reason=done['stop'],checkpoint=done['checkpoint']))
        for name,stats in read(r/'accounts.json').items():
            exits=pd.read_csv(r/'exits.csv'); e=exits[(exits.selector==name)&exits.reason.str.startswith('early_')]
            assert np.isclose(e.paired_delta_usd.sum(),stats['early_minus_hold_usd']) and len(e)==stats['early']
            accounts.append(dict(year=year,arm=name,early_harmful=int((e.paired_delta_usd < -1e-8).sum()),**stats))
    pd.concat(curves).to_csv(out/'curves.csv.gz',index=False)
    pd.DataFrame(accounts).to_csv(out/'accounts.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'mae.csv',index=False)
    (out/'audits.json').write_text(json.dumps(details,indent=2)+'\n')
    chart.savefig(out/'training.png',dpi=110);plt.close(chart)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path);p.add_argument('out',type=Path);p.add_argument('host')
    a=p.parse_args();report(a.root,a.out,a.host)
