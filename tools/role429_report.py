"""Render all completed 429 learning curves and aligned economic endpoints."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from role429_worker import model, heads

OUT=Path('/var/tmp/role429-sell-ranking')


def main():
    torch.set_num_threads(1)
    done=json.loads((OUT/'DONE.json').read_text())
    lines=['# 429: paired ranking results','',
           'Managed advantages in bp relative to waiting, not standalone P&L. '
           'Chronological native accounts use the fixed positive-EV gate. No 2024.','',
           '|H2|Seed|Epochs / stop|Best control/rank|TRAIN bp control/rank|H2 bp control/rank|Random bp|Net % control/rank|Trades control/rank|Corrected/damaged held pairs|',
           '|---|---|---|---|---|---|---|---|---|---|']
    for yr in (2022,2023):
        for seed in (7,19,43):
            tag=f'{yr}-{seed}'
            r=json.loads((OUT/f'{tag}.json').read_text())
            pf=json.loads((OUT/f'{tag}-preflight.json').read_text())
            hist=json.loads((OUT/f'{tag}-history.json').read_text())
            with np.load(OUT/f'{tag}.npz') as z:
                x=torch.from_numpy(z['x']);y=z['y']*pf['scale'];weeks=z['week']
            weights=torch.load(OUT/f'{tag}.pt',map_location='cpu',weights_only=False)['champs']
            fit=[]
            for state in weights:
                n=model();n.load_state_dict(state);n.eval()
                with torch.inference_mode():p=heads(n(x))[-1].numpy()
                picks=[]
                for week in np.unique(weeks):
                    at=np.flatnonzero(weeks==week)
                    best=at[np.argsort(-p[at],axis=0,kind='stable')[:max(1,len(at)//10)]]
                    picks.extend(y[best,np.arange(4)].ravel())
                fit.append(np.mean(picks))
            a,b=r['control'],r['ranking'];c,d=a['account'],b['account']
            lines.append(f"|{yr}|{seed}|{r['epochs']} / {r['stop']}|{r['best_epochs']}|{fit[0]:.2f}/{fit[1]:.2f}|"
                         f"{a['bp']:.2f}/{b['bp']:.2f}|{a['random_bp']:.2f}|{c['ret']:.2f}/{d['ret']:.2f}|"
                         f"{c['trades']}/{d['trades']}|{r['corrected_pairs']}/{r['damaged_pairs']}|")
            fig,axs=plt.subplots(2,1,sharex=True,figsize=(7,5))
            for j,name in enumerate(('control','ranking')):
                for ax,metric in zip(axs,('fit','monitor')):
                    ax.plot([v['epoch'] for v in hist],[v[metric][j] for v in hist],label=name)
                    ax.axvline(r['best_epochs'][j],color=f'C{j}',ls=':',alpha=.65)
                    ax.set_ylabel(metric+' MSE')
            axs[0].legend();axs[1].set_xlabel('Epoch');fig.suptitle(tag);fig.tight_layout()
            fig.savefig(OUT/f'{tag}-learning.png');plt.close(fig)
    lines+=['',f"Decision: {done['decision']} ({done['passed']}/6)."]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
