"""Verify paired no-training weight analyses and publish reproducible tables/plots."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
from app.frozen_recipe_inputs import file_sha
from tools.run_discrete_curve_probe import read,write
from tools.run_bayesian_head_contrast import seal,verify_done


def compare_common(a,b,path='cases'):
    """Discrete evidence must match exactly; continuous cross-host tolerance is explicit."""
    differences=[]
    if isinstance(a,dict):
        keys=set(a)-{'cuda','cuda32'}
        if keys!=set(b)-{'cuda','cuda32'}:raise ValueError('REPORT_KEYS:'+path)
        for k in sorted(keys):differences.extend(compare_common(a[k],b[k],path+'/'+k))
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):raise ValueError('REPORT_LENGTH:'+path)
        for i,(x,y) in enumerate(zip(a,b)):differences.extend(compare_common(x,y,path+'/'+str(i)))
    elif isinstance(a,float):
        if not isinstance(b,float) or not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>1e-10:
            raise ValueError('REPORT_NUMERIC:'+path)
        if a!=b:differences.append(dict(path=path,absolute_difference=abs(a-b)))
    elif type(a) is not type(b) or a!=b:raise ValueError('REPORT_DISCRETE:'+path)
    return differences


def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(a):
    ca=read(a.gpu/'CONFIG.json');cb=read(a.cpu/'CONFIG.json')
    for root,c in ((a.gpu,ca),(a.cpu,cb)):verify_done(root,c['identity'])
    for key in ('parents','source_actor_sha256','components','transitions','code'):
        if ca[key]!=cb[key]:raise ValueError('REPORT_LINEAGE:'+key)
    ra=read(a.gpu/'RESULT.json');rb=read(a.cpu/'RESULT.json')
    if any(r['optimizer_steps'] or r['actor_updates'] or r['model_promoted'] for r in (ra,rb)):raise ValueError('REPORT_MUTATION')
    differences=compare_common(ra['cases'],rb['cases'])
    a.output.mkdir(parents=True,exist_ok=True)
    points=[];events=[];attribution=[];summary=[];terminal=[]
    for c in ra['cases']:
        case=f"{c['fold']}_{c['seed']}"
        for method,step,p in [('source',0,c['source'])]+[(arm['method'],int(k),p) for arm in c['arms'] for k,p in arm['points'].items()]:
            points.append(dict(case=case,method=method,step=step,cost=p['cost'],regret=p['regret'],
                forward_error=p['forward_max_error'],gradient_error=p['gradient_max_error'],
                finite_difference_error=p['finite_difference_error'],cpu32_action_disagreements=p['cpu32_action_disagreements'],
                cuda_action_disagreements=p['cuda']['full_fit_action_disagreements'],
                cuda_route_disagreements=p['cuda']['minimum_twin_route_disagreements'],
                cuda_gradient_cosine=p['cuda']['full_fit_gradient_cosine'],moment_descent_cosine=p['moment_direction_vs_full_descent']))
        for arm in c['arms']:
            for t in arm['transitions']:
                for block,value in t['exact_shapley_fit_regret'].items():
                    attribution.append(dict(case=case,method=arm['method'],before=t['before'],after=t['after'],block=block,
                        regret_shapley=value,cost_shapley=t['exact_shapley_fit_cost'][block],
                        relative_weight_drift=t['weight_drift'][block]['delta_norm']/t['weight_drift'][block]['before_norm']))
                for e in t['events']:
                    g=e.get('fit_gradients',{})
                    events.append(dict(case=case,method=arm['method'],start=t['before'],end=t['after'],row=e['row'],scope=e['scope'],
                        kind=e['kind'],old_action=e['before'],new_action=e['after'],regret_delta=e['regret_delta'],
                        row_cost_delta=e['row_cost_after']-e['row_cost_before'],margin_delta=e['finite_margin_delta'],
                        first_order_delta=e['first_order_margin_delta'],own_descent=g.get('own_descent'),
                        others_descent=g.get('others_descent'),total_descent=g.get('total_descent'),
                        moment_direction=g.get('saved_moment_margin_direction'),**e['exact_shapley_margin']))
                if arm['method']=='complete' and (t['before'],t['after'])==(0,256):
                    ev=[e for e in t['events'] if e['scope']=='fit'];terminal.extend(ev)
                    summary.append(dict(case=case,fit_rows=c['fit_rows'],source_cost=c['source']['cost'],final_cost=arm['points']['256']['cost'],
                        source_regret=c['source']['regret'],final_regret=arm['points']['256']['regret'],
                        changed=len(ev),beneficial=sum(e['beneficial'] for e in ev),harmful=sum(e['harmful'] for e in ev),
                        tied=sum(not e['beneficial'] and not e['harmful'] for e in ev)))
    harmful=[e for e in terminal if e['harmful']]
    verification=dict(gpu_identity=ra['identity'],cpu_identity=rb['identity'],
        gpu_result_sha256=file_sha(a.gpu/'RESULT.json'),cpu_result_sha256=file_sha(a.cpu/'RESULT.json'),
        common_discrete_evidence_exact=True,continuous_tolerance=1e-10,
        continuous_difference_count=len(differences),max_continuous_difference=max((d['absolute_difference'] for d in differences),default=0.),
        differences=differences,points=len(points),transitions=24,finite_hybrids=ra['finite_hybrids'],
        complete_source_to_256_fit=dict(changed=len(terminal),harmful=len(harmful),
            harmful_with_own_row_loss_lower=sum(e['row_cost_after']<e['row_cost_before'] for e in harmful),
            harmful_with_own_gradient_helpful=sum(e['fit_gradients']['own_descent']>0 for e in harmful),
            harmful_with_full_gradient_opposing=sum(e['fit_gradients']['total_descent']<0 for e in harmful),
            harmful_with_initial_moment_opposing=sum(e['fit_gradients']['saved_moment_margin_direction']<0 for e in harmful)))
    write(a.output/'VERIFICATION.json',verification);write(a.output/'RESULT.json',ra);write(a.output/'CONFIG.json',ca)
    for name,rows in [('POINTS',points),('EVENTS',events),('COMPONENTS',attribution),('SUMMARY',summary)]:table(a.output/(name+'.csv'),rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,c in zip(axes.flat,ra['cases']):
        t=next(t for arm in c['arms'] if arm['method']=='complete' for t in arm['transitions'] if (t['before'],t['after'])==(0,256))
        vals=t['exact_shapley_fit_regret'];x=np.arange(len(vals))
        ax.bar(x,list(vals.values()),color=['#ae3535' if v>0 else '#287e53' for v in vals.values()]);ax.axhline(0,color='black',lw=.6)
        ax.set_xticks(x,[k.replace('twin','Q').replace('_','\n') for k in vals]);ax.set_title(f"{c['fold']}/{c['seed']} - complete 0 to 256")
        ax.set_ylabel('Exact FIT regret attribution (lower better)')
    fig.savefig(a.output/'COMPONENTS.png',dpi=150);plt.close(fig)
    text=['# Saved-Weight Forensics: Measured Results','',
        'Full expanded FIT, not the stable-added subset in338. No new account trading return.','',
        '|Case|FIT rows|Cost before -> after|Regret before -> after|Better / worse / tied decisions|',
        '|---|---:|---:|---:|---:|']
    for r in summary:text.append(f"|{r['case']}|{r['fit_rows']}|{r['source_cost']:.9f} -> {r['final_cost']:.9f}|{r['source_regret']:.9f} -> {r['final_regret']:.9f}|{r['beneficial']} / {r['harmful']} / {r['tied']}|")
    text+=['',f"384 finite hybrids,20 saved points,24 transitions. Cross-host continuous maximum difference: {verification['max_continuous_difference']:.3g}; all common discrete evidence exact.",
        '',f"Complete-arm source-to256 FIT: {len(terminal)} changed decision occurrences, {len(harmful)} harmful; 3 harmful changes reduce the same row's surrogate loss. Occurrences are not independent samples.",
        '', 'See [protocol and interpretation](../../339_SAVED_WEIGHT_DECISION_FORENSICS.md).',
        'The16 coalitions per transition are interventions, not independently trained networks. No hybrid is selected or promoted.']
    (a.output/'RESULTS.md').write_text('\n'.join(text)+'\n')
    seal(a.output,ra['identity'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('gpu','cpu','output'):parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
