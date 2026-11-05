"""Bounded, CPU-only diagnostics from existing full-account epoch evaluations."""
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def atomic_bytes(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


def flatten(data, prefix=''):
    out = {}
    for key, value in data.items():
        name = prefix+key
        if isinstance(value, dict):
            out.update(flatten(value, name+'.'))
        elif value is None or isinstance(value, (int, float, str, bool)):
            out[name] = value if not isinstance(value, float) or math.isfinite(value) else None
    return out


def render_diagnostics(history, directory, design, best_epoch=None):
    if not history or any(a['epoch'] >= b['epoch'] for a,b in zip(history,history[1:])):
        raise ValueError('EPOCH_ORDER')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    epochs = [r['epoch'] for r in history]
    warmup = design.get('monitor_start_epoch', 1)
    if best_epoch is None:
        best_epoch = next((r['epoch'] for r in reversed(history) if r['monitor'].get('improved')), None)
    rows = []
    for row in history:
        record = {k:row.get(k) for k in ('epoch','steps','updates','training_seconds','evaluation_seconds')}
        record.update(flatten(row.get('metrics', {})))
        record.update(flatten(row.get('monitor', {}), 'monitor.'))
        record.update(flatten(row.get('last_update', {}), 'last_update.'))
        rows.append(record)
    fields = list(rows[0]) + sorted(set().union(*(r.keys() for r in rows))-set(rows[0]))
    stream = io.StringIO(newline=''); writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    atomic_bytes(directory/'EPOCHS.csv', stream.getvalue().encode())

    def series(path):
        return [r[path] if finite(r.get(path)) else float('nan') for r in rows]

    def decorate(ax):
        if warmup > 1:
            ax.axvspan(.5, min(warmup, epochs[-1])+.5, color='#dddddd', alpha=.45, label='Warmup')
        if best_epoch is not None:
            ax.axvline(best_epoch, color='#b23445', linestyle='--', linewidth=1, label='Selected checkpoint')
        ax.set_xlim(.5, max(epochs)+.5); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)

    def save(fig, name):
        FigureCanvasAgg(fig)
        output = io.BytesIO(); fig.savefig(output, format='png', dpi=120)
        atomic_bytes(directory/name, output.getvalue()); fig.clear()

    fig = Figure(figsize=(11, 10), layout='constrained')
    axes = fig.subplots(4, 1)
    for ax, split, color in zip(axes[:2], ('TRAIN','VALIDATION'), ('#187977','#bb4958')):
        ax.plot(epochs, series(split+'.return_pct'), color=color)
        ax.set_ylabel('Net return (%)'); ax.set_title(split+' full account (independent y-axis)'); decorate(ax)
    for split, color in (('TRAIN','#187977'), ('VALIDATION','#bb4958')):
        axes[2].plot(epochs,series(split+'.return_per_week'),label=split,color=color)
    axes[2].set_ylabel('Return / week (percentage points)'); decorate(axes[2]); axes[2].legend()
    axes[3].plot(epochs,series('monitor.score'),label='Stopping score',color='#bb4958')
    axes[3].plot(epochs,series('monitor.best'),label='Best accepted',color='#333333',linestyle=':')
    axes[3].set_ylabel('Stopping metric'); decorate(axes[3]); axes[3].legend()
    fig.suptitle('Training and validation | selected epoch: '+str(best_epoch))
    save(fig,'TRAIN_VALIDATION.png')

    fig = Figure(figsize=(12, 10), layout='constrained'); axes = list(fig.subplots(3,2).flat)
    for ax, metric, title in zip(axes[:4], ('sharpe','max_drawdown_pct','orders_per_week','win_rate'),
                                ('Sharpe','Maximum drawdown (%)','Orders / week','Win rate (fraction)')):
        for split, color in (('TRAIN','#187977'), ('VALIDATION','#bb4958')):
            ax.plot(epochs,series(split+'.'+metric),label=split,color=color)
        ax.set_title(title); decorate(ax); ax.legend()
    ax=axes[4]; ax.step(epochs,series('monitor.bad'),where='mid',color='#555555')
    if 'patience' in design: ax.axhline(design['patience'],color='#bb4958',linestyle=':')
    ax.set_title('Patience counter'); decorate(ax)
    ax=axes[5]
    for key in ('training_seconds','evaluation_seconds'):
        ax.plot(epochs,series(key),label=key)
    ax.set_title('Seconds per epoch'); decorate(ax); ax.legend()
    fig.suptitle('Training diagnostics | missing metrics remain missing')
    save(fig,'TRAINING_DIAGNOSTICS.png')
    files=['TRAIN_VALIDATION.png','TRAINING_DIAGNOSTICS.png','EPOCHS.csv']
    if any(row.get('last_update') for row in history):
        fig=Figure(figsize=(12,8),layout='constrained'); axes=list(fig.subplots(2,3).flat)
        for ax,key in zip(axes[:5],('actor','critic','gradient_norm','alpha','target_entropy')):
            ax.plot(epochs,series('last_update.'+key),color='#187977')
            ax.set_title(key+' (last update of epoch)'); decorate(ax)
        gap=[a-b if finite(a) and finite(b) else float('nan') for a,b in
             zip(series('TRAIN.return_per_week'),series('VALIDATION.return_per_week'))]
        axes[5].plot(epochs,gap,color='#bb4958'); axes[5].set_title('TRAIN - VAL return/week'); decorate(axes[5])
        fig.suptitle('Optimizer diagnostics | last-update snapshots, not epoch averages')
        save(fig,'OPTIMIZER_DIAGNOSTICS.png'); files.append('OPTIMIZER_DIAGNOSTICS.png')
    result=dict(epochs=len(rows),best_epoch=best_epoch,
        history_sha256=hashlib.sha256(json.dumps(history,sort_keys=True).encode()).hexdigest(),
        source='existing full TRAIN/VALIDATION evaluations; no additional inference',
        warmup=warmup,select_during_warmup=design.get('select_during_warmup',False),
        files=files)
    atomic_bytes(directory/'DIAGNOSTICS.json', (json.dumps(result,indent=2,sort_keys=True)+'\n').encode())
    return result
