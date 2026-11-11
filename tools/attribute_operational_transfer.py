"""Attribute saved operational actor interventions using exact native continuations."""
import argparse
from pathlib import Path
import time

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,project_inputs,support_hash
from app.structural_refit import verify_files
from app.three_signal_learning import environment
from tools.attribute_ranking_improvement import attribute_pair
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now,configure
from tools.run_structural_sac_refit import market_for


def run(a):
    configure(7)
    started=time.monotonic()
    cfg=read(a.source/'CONFIG.json');verify_done(a.source,cfg['identity'])
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    recipe=read(a.bundle/'RECIPE.json')
    if parent['identity']!=cfg['parent_identity'] or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:
        raise ValueError('ATTRIBUTION_PARENT')
    market=market_for(a.parent,parent)
    market.inputs=project_inputs(market.inputs,recipe);market.recipe=recipe
    if support_hash(market.inputs)!=cfg['support_sha256'] or set(recipe['split']['periods'])!={'TRAIN','VALIDATION'}:
        raise ValueError('ATTRIBUTION_SUPPORT')
    contract=dict(source=cfg['identity'],recipe=recipe['identity'],updates=[1,32],
        seeds=cfg['seeds'],arms=cfg['arms'],code=file_sha(__file__),training=False,promotion=False)
    identity=digest(contract)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=contract:
        raise ValueError('ATTRIBUTION_CONFIG')
    write(a.output/'CONFIG.json',contract)
    risk=read(a.parent/'RISK.json');records=[]
    for steps in contract['updates']:
        for arm in contract['arms']:
            for seed in contract['seeds']:
                for period,interval in recipe['split']['periods'].items():
                    key=f'{arm}_{seed}_{steps}_{period}'
                    cell=a.output/key
                    if (cell/'DONE.json').exists():
                        verify_done(cell,identity);records.append(read(cell/'SUMMARY.json'));continue
                    def progress(**kw):
                        write(a.output/'PROGRESS.json',dict(stage=key,at=now(),seconds=time.monotonic()-started,**kw))
                    def make():
                        return environment(market,risk,parent['design']['horizon'],interval,cfg['effective']['gamma'])
                    progress(events=0)
                    result=attribute_pair(make,market.inputs.row_ts_ns,a.source/'reference'/period,
                        a.source/f'{arm}_{seed}'/f'actor_{steps:04d}'/period,progress)
                    summary=dict(arm=arm,seed=seed,updates=steps,period=period,
                        **{k:v for k,v in result.items() if k!='events'})
                    summary['harmful_changes']=sum(e['contribution_usd'] < -1e-6 for e in result['events'])
                    summary['helpful_changes']=sum(e['contribution_usd'] > 1e-6 for e in result['events'])
                    write(cell/'EVENTS.json',result['events']);write(cell/'SUMMARY.json',summary)
                    seal(cell,identity);records.append(summary)
    write(a.output/'RESULT.json',dict(identity=identity,accounts=records,training=False,
        model_promoted=False,finished_at=now(),seconds=time.monotonic()-started))
    seal(a.output,identity)
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),accounts=len(records)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','bundle','output'):
        p.add_argument('--'+name,required=True,type=Path)
    run(p.parse_args())
