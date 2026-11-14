"""Publish sealed crossing witnesses as a compact stage-attribution report."""
import argparse
import csv
from pathlib import Path
import shutil
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write


def run(a):
    config=read(a.verified/'CONFIG.json');verify_done(a.verified,config['identity'])
    result=read(a.verified/'RESULT.json');verification=read(a.verified/'VERIFICATION.json')
    with (a.verified/'EVENTS.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=verification['events']:raise ValueError('CROSS_REPORT_COUNT')
    a.output.mkdir(parents=True,exist_ok=True)
    for name in ('CONFIG.json','RESULT.json','VERIFICATION.json','EVENTS.csv'):shutil.copyfile(a.verified/name,a.output/name)
    names={0:'NOP',1:'BUY',2:'SELL'}
    lines=['# First Harmful Crossing Results','',
        'Authenticated replay, not new account trading performance. All rows are FIT.',
        'Each row is its first harmful transition in a case; rows can recur across cases.',
        '', '|Case|Step|Row|Before -> proposed -> accepted|In batch|Pair margin before / proposed / accepted|',
        '|---|---:|---:|---|---:|---:|']
    for r in rows:
        actions=' -> '.join(names[int(r[k])] for k in ('before','proposed','after'))
        margins=' / '.join(f"{float(r[k]):+.8f}" for k in ('margin_before','margin_proposed','margin_after'))
        lines.append(f"|{r['case']}|{r['offset']}|{r['row']}|{actions}|{r['batch_multiplicity']}|{margins}|")
    lines+=['',f"{verification['replayed_transactions']} transactions and {verification['original_endpoints_exact']} original endpoints exactly reproduced; {verification['cold_witnesses_exact']} cold witness replays exact.",
        f"Independent Torch64 derivative maximum difference: {verification['derivative_max_error']:.3g}.",
        '', 'Finite margin contributions sum exactly across stages. Positive protects the old action against the accepted harmful action; negative weakens it.',
        'The raw proposal is a counterfactual branch, not a valid replacement for the protected update: it may violate the original guard.',
        '', '[Interpretation and next action](../../340_FIRST_HARMFUL_CROSSING_REPLAY.md).']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    ax.bar(x-.18,[float(r['finite_adam']) for r in rows],width=.36,label='Raw AdamW proposal',color='#3971ac')
    ax.bar(x+.18,[float(r['finite_repair']) for r in rows],width=.36,label='Finite protective correction',color='#b75e31')
    ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Change in old-minus-harmful action margin')
    ax.set_xticks(x,[f"{r['case']}\nstep {r['offset']}\nrow {r['row']}" for r in rows],fontsize=8)
    ax.legend();fig.savefig(a.output/'STAGES.png',dpi=150);plt.close(fig)
    write(a.output/'REPORT_IDENTITY.json',dict(source_identity=config['identity'],events=len(rows),model_promoted=False))
    seal(a.output,config['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verified',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
