"""Authenticated CPU preflight of nested-history supports and producer scopes."""
import argparse
import os
from pathlib import Path

from app.nested_history_preflight import supports,contained
from app.frozen_recipe_inputs import file_sha
from app.structural_refit import verify_files
from app.selected_input_market import market_for
from app.g3_producer_audit import dependency_inventory
from app.rl_subset import support_hash
from tools.run_prioritized_training_diagnosis import read,write


def run(config,output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('FORCE_CPU')
    parent=Path(config['parent']);bundle=Path(config['bundle'])
    spec=read(parent/'SPEC.json');recipe=read(bundle/'RECIPE.json');manifest=read(parent/'data/manifest.json')
    verify_files(bundle,read(bundle/'MANIFEST.json')['files'])
    verify_files(parent,{k:v for k,v in spec['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    if file_sha(bundle/'model.pt')!=config['source']:raise ValueError('SOURCE_IDENTITY')
    inputs=market_for(parent,spec,recipe).inputs
    if support_hash(inputs)!=config['support']:raise ValueError('SUPPORT_IDENTITY')
    receipt_path=Path(__file__).parents[1]/'docs/work_plan/generalization/g3_producer354_receipt.json'
    producer=read(receipt_path)
    inventory=dependency_inventory(recipe['list']['columns'],[x['name'] for x in producer['activity_predictors']])
    scoped=set(manifest['consumer_scope']['changed_columns'])
    learned=[x for x in inventory['columns'] if any(d.startswith('FROZEN_') for d in x['primitive_dependencies'])]
    arms=supports(inputs.row_ts_ns,inputs.slot_ts_ns,inputs.valid)
    for arm in arms:
        checks=[]
        for col in learned:
            fit=manifest['consumer_scope']['fit_window'] if col['column'] in scoped else None
            checks.append(dict(column=col['column'],producer=col['producer'],known_fit=fit,
                contained=contained(fit,arm['interval']),
                action='REFIT_WITHIN_SUBSET_AND_REBUILD' if fit else 'RECOVER_FACTOR_FIT_SCOPE_THEN_REFIT'))
        arm.update(learned_columns=checks,scaler='REFIT_FIXED_TRAIN_ON_SUBSET',
            network='FRESH_ALL_COMPONENTS_AND_OPTIMIZER_REPLAY',training_eligible=False,
            reason='PREVIEW_SUPPORT_ONLY_PRODUCERS_NOT_REFITTED')
    write(output,dict(schema='nested_history371.v1',source=config['source'],support=config['support'],
        producer_receipt_sha256=file_sha(receipt_path),columns=len(recipe['list']['columns']),
        groups=len(recipe['representation']['resolved']['indices']),window=recipe['tensor']['window'],
        learned_column_count=len(learned),arms=arms,seeds=[7,19,43],training_started=False,
        exact_next_action='Recover factor fitting range and refit activity/GMM/Kalman within each subset; then scaler and fresh networks',
        note='Counts on current store only; no claim that new producer values or supports already exist'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(read(a.config),a.output)
