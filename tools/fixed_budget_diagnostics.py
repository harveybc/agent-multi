"""Cheap checkpoint plots from authenticated frozen-policy reports, no fits."""
import json
import os
from pathlib import Path
import tempfile

from tools import paired_checkpoint_diagnostics as diagnostic


def refresh(root):
    root=Path(root)
    if (root/'CLOSE.json').exists():return dict(state='CLOSED_IMMUTABLE')
    paths=sorted(root.glob('*/*/paired-monitor/step-*/PAIR.json'))
    if not paths:return dict(state='WAIT_COMPLETE_PAIR')
    if len({p.parent.parent for p in paths})!=1:
        raise ValueError('FIXED_PLOT_MULTIPLE_ATTEMPTS')
    output=paths[0].parent.parent/'diagnostics'
    source_pins={p.parent.name:diagnostic.digest(p) for p in paths}
    status=output/'STATUS.json'
    if status.exists() and diagnostic.read(status).get('source_pairs')==source_pins:
        for name,pin in diagnostic.read(status)['files'].items():
            if diagnostic.digest(output/name)!=pin:raise ValueError('FIXED_PLOT_CHANGED')
        return dict(state='UNCHANGED_AUTHENTICATED_PAIRS')
    result,_,_=diagnostic.analyze(root)
    if result['settings'].get('early_stopping') is not False:
        raise ValueError('FIXED_PLOT_PROTOCOL_REQUIRED')
    quarters=[]
    for path,row in zip(paths,result['rows']):
        pair=diagnostic.read(path);metrics={}
        for task in pair['reports']:
            summary=diagnostic.read(task['summary'])
            metrics[task['split']]={k:summary.get(k) for k in (
                'net_profit','total_return','max_drawdown_pct','sharpe_ratio','trades_closed','win_rate_pct','profit_factor')}
        quarters.append(dict(step=row['step'],metrics=metrics))
    result['trading_checkpoints']=quarters
    result['interpretation']='DESCRIPTIVE_CURVES_NO_METRIC_DRIVEN_STOP_OR_CHECKPOINT_SELECTION'
    diagnostic.write(output/'CURVES.json',result)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    steps=[r['step'] for r in result['rows']]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for role,color in [('train','#136f63'),('validation','#bd4328')]:
        axes[0,0].plot(steps,[100*r[role+'_growth'] for r in result['rows']],marker='o',label=role.upper(),color=color)
    axes[0,0].set_ylabel('Mean daily log growth (%)');axes[0,0].legend()
    axes[0,1].plot(steps,[100*r['gap'] for r in result['rows']],marker='o',label='TRAIN - VALIDATION',color='#7159a0')
    axes[0,1].axhline(0,color='gray',linestyle='--');axes[0,1].set_ylabel('Daily log-growth gap (percentage points)');axes[0,1].legend()
    axes[1,0].plot(steps,[100*r['metrics']['validation']['total_return'] for r in quarters],marker='o',label='VALIDATION net return')
    axes[1,0].plot(steps,[r['metrics']['validation']['max_drawdown_pct'] for r in quarters],marker='s',label='VALIDATION max drawdown')
    axes[1,0].set_ylabel('Percent');axes[1,0].legend()
    axes[1,1].plot(steps,[r['metrics']['validation']['trades_closed'] for r in quarters],marker='o',label='VALIDATION closed trades')
    axes[1,1].set_ylabel('Count');axes[1,1].legend()
    for ax in axes.flat:ax.set_xlabel('Training transitions');ax.grid(alpha=.2)
    fig.suptitle('Fixed budget - full TRAIN 2020-23 / VALIDATION 2024 - final model 100000')
    output.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(dir=output,suffix='.png');os.close(fd)
    try:
        fig.savefig(temp,format='png',dpi=120)
        with open(temp,'rb') as stream:os.fsync(stream.fileno())
        os.replace(temp,output/'CURVES.png')
    finally:
        plt.close(fig)
        if os.path.exists(temp):os.unlink(temp)
    diagnostic.write(status,dict(state='PLOTS_AND_CURVES_READY',source_pairs=source_pins,
        files={name:diagnostic.digest(output/name) for name in ('CURVES.json','CURVES.png')}))
    return dict(state='PLOTS_AND_CURVES_READY',pairs=len(paths))
