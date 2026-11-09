"""Wait for the bounded contrast and export authenticated tables and plots."""
import argparse
import csv
from pathlib import Path
import time

from app.frozen_recipe_inputs import file_sha
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import read,write,now


def trading_rows(result):
    rows=[]
    for arm,value in result['arms'].items():
        for point in value['accounts']:
            for period,m in point['metrics'].items():
                row=dict(arm=arm,actor_updates=point['updates'],period=period,eligible=point['eligible'])
                for name in ('net_profit_usd','return_pct','sharpe','max_drawdown_pct','trades','win_rate','orders_per_week'):
                    row[name]=m[name]
                for side in ('buy','sell'):
                    row[side+'_trades']=m[side]['trades']
                    row[side+'_win_rate']=m[side]['wins']/m[side]['trades'] if m[side]['trades'] else None
                row['early_closes']=m['close_reasons'].get('AGENT_CLOSE',0)
                rows.append(row)
    return rows


def export(root,output):
    receipt=verify_collection(root/'fit');result=read(root/'fit/RESULT.json')
    output.mkdir(parents=True,exist_ok=True)
    rows=trading_rows(result)
    with (output/'TRADING.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    calibration=[]
    for arm,value in result['arms'].items():
        for point in value['calibration']:
            for role in ('fit','monitor'):
                for target in ('greedy','stochastic'):
                    metrics=point['metrics'][role]['all'][target]
                    calibration.append(dict(arm=arm,updates=point['updates'],role=role,target=target,
                        **{k:metrics[k] for k in ('rmse','pairwise_accuracy','mean_regret','pairwise_comparisons')}))
    with (output/'CALIBRATION.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(calibration[0]));writer.writeheader();writer.writerows(calibration)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,(role,target) in zip(axes.flat,[(r,t) for r in ('fit','monitor') for t in ('stochastic','greedy')]):
        for arm in result['arms']:
            selected=[r for r in calibration if r['arm']==arm and r['role']==role and r['target']==target]
            x=[r['updates'] for r in selected];y=[r['mean_regret'] for r in selected]
            if len(x)==1:ax.axhline(y[0],label=arm,linestyle='--')
            else:ax.plot(x,y,marker='.',label=arm)
        ax.set(title=f'{role} anchors / {target} continuation',xlabel='Critic updates',ylabel='Mean action regret (lower is better)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'ACTION_REGRET.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,period in zip(axes,('TRAIN','VALIDATION')):
        for arm in result['arms']:
            selected=[r for r in rows if r['arm']==arm and r['period']==period]
            ax.plot([r['actor_updates'] for r in selected],[r['return_pct'] for r in selected],marker='o',label=arm)
        ax.set(title=period,xlabel='Actor updates',ylabel='Full account return (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'TRADING_RETURNS.png',dpi=140);plt.close(fig)
    write(output/'REPORT.json',dict(source_identity=receipt['identity'],generated_at=now(),
        fit_done_sha256=file_sha(root/'fit/DONE.json'),verified_files=len(receipt['files']),
        producer_sha256=file_sha(Path(__file__)),model_promoted=False,
        files={n:file_sha(output/n) for n in ('TRADING.csv','CALIBRATION.csv','ACTION_REGRET.png','TRADING_RETURNS.png')}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();start=time.monotonic()
    while not (a.root/'QUEUE_DONE.json').exists():
        if not a.wait:raise ValueError('RANKING_QUEUE_NOT_COMPLETE')
        if list(a.root.glob('FAILED_*.json')):raise ValueError('RANKING_QUEUE_FAILED')
        if time.monotonic()-start>86400:raise TimeoutError('RANKING_REPORT_WAIT_LIMIT')
        time.sleep(15)
    export(a.root,a.root/'reports')


if __name__=='__main__':main()
