"""Summarize already fitted 490 probes; pair weeks and average seeds first."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(root):
    panels=[json.loads((root/f'role{r}.json').read_text()) for r in (0,1,-1)]
    df=pd.DataFrame(sum(panels,[]))
    keys=['role','first','last','later','seed','control','part']
    assert len(df)==216 and not df.duplicated(keys).any()
    rows=[]
    for key,frame in df.groupby(keys[:4]+['part']):
        record=dict(zip(keys[:4]+['part'],key))
        week={}
        for control,block in frame.groupby('control'):
            week[control]=pd.DataFrame(block.by_week.tolist()).mean(axis=0).sort_index()
            record['n']=int(block.n.iloc[0])
            record['informative_real']=int(frame[frame.control=='true'].n_informative.iloc[0])
            record[control+'_accuracy']=float(block.best_rate.mean())
            if control!='planted':record[control+'_usd_vs_actor']=float(week[control].mean())
        diff=(week['true']-week['null']).to_numpy()
        assert np.isfinite(diff).all()
        boot=np.random.default_rng(490).choice(diff,(5000,len(diff))).mean(1)
        record.update(weeks=len(diff),true_minus_null=float(diff.mean()),
                      ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)))
        rows.append(record)
    pd.DataFrame(rows).to_csv(root/'summary.csv',index=False,float_format='%.6f')


if __name__=='__main__':summarize(Path(sys.argv[1]))
