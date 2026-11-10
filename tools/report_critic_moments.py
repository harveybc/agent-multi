"""Authenticate and compare paired critic objectives with Bayesian actors."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.structural_refit import verify_files
from tools.run_discrete_curve_probe import read, write, now


def authenticated(root):
    config=read(root/'CONFIG.json');done=read(root/'DONE.json')
    if config['identity']!=done['identity']:raise ValueError('MOMENT_REPORT_IDENTITY')
    verify_files(root,done['files'])
    return config


def validate_pair(control, candidate):
    for k in ('source_sha256','manifest_sha256','parent_identity','support_sha256',
              'recipe_identity','torch','device'):
        if control[k]!=candidate[k]:raise ValueError('MOMENT_PAIR:'+k)
    for k,v in control['design'].items():
        if k!='arms' and candidate['design'].get(k)!=v:
            raise ValueError('MOMENT_PAIR_DESIGN:'+k)
    if candidate['design']['arms']!=['bayesian']:
        raise ValueError('MOMENT_PAIR_ARMS')


def error_summary(value):
    return dict(bias=value['bias'],mae=value['mae'],rmse=value['rmse'],
        residual_std=math.sqrt(max(0,value['rmse']**2-value['bias']**2)),n=value['n'])


def report(control, diagnostics, candidates, output):
    base=authenticated(control);diag=authenticated(diagnostics)
    if diag.get('control_diagnostics_identity')!=base['identity']:
        raise ValueError('MOMENT_CONTROL_DIAGNOSTIC_PROVENANCE')
    rows=[];errors={};identities={}
    roots={'smooth_l1':control,**candidates}
    for name,root in roots.items():
        config=authenticated(root);identities[name]=config['identity']
        if name!='smooth_l1':
            validate_pair(base,config)
            if config['design']['critic_objective']!=name:
                raise ValueError('MOMENT_LABEL')
            expected_weight={'mean_var_1':1.,'mean_var_05':.5}[name]
            if config['design']['variance_weight']!=expected_weight:
                raise ValueError('MOMENT_VARIANCE_WEIGHT')
        if read(root/'REPLAY.json')!=read(control/'REPLAY.json'):
            raise ValueError('MOMENT_REPLAY_CHANGED')
        error_path=(diagnostics if name=='smooth_l1' else root)/'common/ERRORS.json'
        e=read(error_path)
        errors[name]={p:error_summary(e[p]) for p in ('TRAIN','VALIDATION')}
        for seed in base['design']['seeds']:
            r=read(root/f'bayesian_{seed}/RESULT.json')
            reference=read(control/f'bayesian_{seed}/RESULT.json')
            if r['schedule_sha256']!=reference['schedule_sha256']:
                raise ValueError('MOMENT_ACTOR_SCHEDULE')
            initial=torch.load(root/f'bayesian_{seed}/update_0000/HEAD.pt',map_location='cpu',weights_only=False)['head']
            initial_ref=torch.load(control/f'bayesian_{seed}/update_0000/HEAD.pt',map_location='cpu',weights_only=False)['head']
            if set(initial)!=set(initial_ref) or any(not torch.equal(v,initial_ref[k]) for k,v in initial.items()):
                raise ValueError('MOMENT_ACTOR_INITIALIZATION')
            point=r['points'][-1]
            if point['updates']!=256:raise ValueError('MOMENT_ENDPOINT')
            head=torch.load(root/f'bayesian_{seed}/update_0256/HEAD.pt',map_location='cpu',weights_only=False)['head']
            std=torch.cat([torch.nn.functional.softplus(head[k]).flatten()
                           for k in ('weight_rho','bias_rho')])
            rows.append(dict(objective=name,seed=seed,metrics=point['metrics'],
                uncertainty=point['uncertainty'],eligible=point['eligible'],
                posterior_weight_std_mean=float(std.mean()),posterior_weight_std_min=float(std.min()),
                posterior_weight_std_max=float(std.max()),
                independent_draw_action_disagreements=r['draw_action_disagreements']))
    means={}
    for name in roots:
        selected=[r for r in rows if r['objective']==name]
        means[name]=dict(train_return_pct=float(np.mean([r['metrics']['TRAIN']['return_pct'] for r in selected])),
            validation_return_pct=float(np.mean([r['metrics']['VALIDATION']['return_pct'] for r in selected])),
            validation_probability_std=float(np.mean([r['uncertainty']['VALIDATION']['probability_std'] for r in selected])),
            all_eligible=all(r['eligible'] for r in selected))
    output.mkdir(parents=True,exist_ok=True)
    write(output/'SUMMARY.json',dict(identities=identities,rows=rows,means=means,critic_errors=errors,
        pairing_verified=True,model_promoted=False,uncertainty_calibrated=False,
        note='Shared critic calibration seed and inherited representation; three actor refinement seeds, not three independently trained critics.'))
    lines=['# Critic moments with Bayesian actor','','All arms use the same Bayesian actor design. Primary endpoint256.',
        'Uncertainty is probability dispersion, not accuracy or calibrated confidence.','',
        '| Loss | Seed | TRAIN % | VAL % | VAL Sharpe | VAL DD % | VAL trades | VAL win rate | VAL probability std |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        t,v=r['metrics']['TRAIN'],r['metrics']['VALIDATION']
        lines.append(f"| {r['objective']} | {r['seed']} | {t['return_pct']:.6f} | {v['return_pct']:.6f} | {v['sharpe']} | {v['max_drawdown_pct']:.6f} | {v['trades']} | {v['win_rate']} | {r['uncertainty']['VALIDATION']['probability_std']:.9f} |")
    lines+=['','## Critic return errors','','| Loss | Period | Bias | Residual std | MAE | RMSE |','|---|---|---:|---:|---:|---:|']
    for name,periods in errors.items():
        for p,e in periods.items():lines.append(f"| {name} | {p} | {e['bias']:.6f} | {e['residual_std']:.6f} | {e['mae']:.6f} | {e['rmse']:.6f} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    names=list(means)
    axes[0].bar(names,[means[n]['validation_return_pct'] for n in names]);axes[0].set_ylabel('Mean validation return %')
    axes[1].bar(names,[means[n]['validation_probability_std'] for n in names]);axes[1].set_ylabel('Mean posterior probability std')
    fig.tight_layout();fig.savefig(output/'COMPARISON.png');plt.close(fig)
    write(output/'DONE.json',dict(at=now(),identities=identities,files={p.name:file_sha(p)
        for p in output.iterdir() if p.is_file() and p.name!='DONE.json'}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('control','diagnostics','mse','half','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();report(a.control,a.diagnostics,{'mean_var_1':a.mse,'mean_var_05':a.half},a.output)


if __name__=='__main__':main()
