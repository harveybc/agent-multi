"""Report all feature tails against real quarterly variation before training."""
import argparse
import json
from pathlib import Path
import pandas as pd


def inspect(root, spec):
    cols = list(pd.read_parquet(root/'context7/FEATURES71.parquet').columns)
    real = pd.read_parquet(json.loads(spec.read_text())['producer_settings']['store'],
        columns=['timestamp',*cols],filters=[('timestamp','>=',pd.Timestamp('2020',tz='UTC')),
        ('timestamp','<',pd.Timestamp('2024',tz='UTC'))]).set_index('timestamp')
    fit = real[real.index.year<=2021]
    lo,hi = fit.quantile(.01),fit.quantile(.99)
    rows = []
    for folder in sorted(p for p in root.iterdir() if (p/'OHLC.parquet').exists()):
        files = sorted(folder.glob('FEATURES*.parquet'))
        f = pd.read_parquet(files[-1])
        for col in f.columns:
            def tails(s):
                return ((s<lo[col])|(s>hi[col])).where(s.notna())
            ref = real.loc[real.index.year>=2022,col]
            tail = tails(ref).astype(float)
            quarters = tail.groupby([tail.index.year,tail.index.quarter]).mean()
            s = tails(f[col]).mean()
            rows.append((folder.name,col,s,quarters.min(),quarters.max(),s>quarters.max()))
    df = pd.DataFrame(rows,columns=['path','feature','outside','real_quarter_min','real_quarter_max','above_all'])
    df.to_csv(root/'INPUT_TAILS.csv',index=False)
    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args()
    result=inspect(args.root,args.spec)
    print(result.groupby('path').above_all.sum().to_string())
