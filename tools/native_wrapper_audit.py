"""Incremental CPU audit of completed native-wrapper receipts and frozen policies."""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from app.frozen_recipe_inputs import file_sha, require
from app.rl_subset import digest, subset_recipe, project_inputs, support_hash
from app.native_policy_sensitivity import probe_policy, FixedRule
from tools.recipe_sac_queue import read, write, now, software, tensor_hash, evaluate


def local_artifact(path, original_root, mirror):
    relative=Path(path).relative_to(original_root)
    result=(mirror/relative).resolve()
    require(result.is_relative_to(mirror.resolve()),'MIRROR_PATH_ESCAPE')
    return result


def ledger_checks(report,trades):
    require(len(trades)==report['trades'],'LEDGER_COUNT')
    def equal(a,b): return np.isclose(a,b,rtol=1e-10,atol=1e-7)
    require(equal(sum(t['net_usd'] for t in trades),report['net_profit_usd']),'LEDGER_PROFIT')
    for key in ('spread','slippage','commission','financing'):
        require(equal(sum(t[key+'_usd'] for t in trades),report['cost_'+key+'_usd']),'LEDGER_COST:'+key)
    for t in trades:
        require(t['direction'] in (-1,1),'ORDER_DIRECTION')
        numeric=('units','percentual_order_size','stop_loss','entry_price','take_profit',
                 'gross_usd','net_usd','spread_usd','slippage_usd','commission_usd','financing_usd')
        require(all(np.isfinite(t[k]) for k in numeric),'ORDER_NONFINITE')
        require(t['units']>0 and 0<t['percentual_order_size']<=1,'ORDER_VOLUME')
        require((t['stop_loss']<t['entry_price']<t['take_profit']) if t['direction']==1 else
                (t['take_profit']<t['entry_price']<t['stop_loss']),'ORDER_PROTECTION')
        require(equal(t['gross_usd']-sum(t[k+'_usd'] for k in ('spread','slippage','commission','financing')),
                      t['net_usd']),'TRADE_NET')
        require(t['entry_time']<=t['exit_time'],'TRADE_CLOCK')
    return dict(state='PASS',trades=len(trades),costs_reconciled=True,sl_tp_volume_present=True)


def run(a):
    import torch
    from stable_baselines3 import PPO
    from feature_selector.evaluators import RLModelEvaluator
    from tools.rl_subset_bridge import SubsetMarket
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(not a.output.resolve().is_relative_to(a.root.resolve()),'ISOLATED_OUTPUT')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'AUDIT.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=read(a.root/'SPEC.private.json')
    require(file_sha(a.root/'SPEC.private.json')==read(a.root/'IDENTITY.json')['spec_sha256'],'SPEC_PIN')
    require(spec['software']==software(),'SOFTWARE_PIN')
    original_root=Path(spec['inputs']['recipe']).parent
    runtime=Path(__file__).resolve().parents[1]
    for role,base in [('agent',runtime),('engine',a.engine_root),('selector',a.selector_root)]:
        for name,sha in spec['code'][role].items():
            require(file_sha(base/name)==sha,'IMPORTED_SOURCE_PIN:'+role+':'+name)
    identity=dict(spec_sha256=file_sha(a.root/'SPEC.private.json'),tool_sha256=file_sha(Path(__file__)),
        probe_sha256=file_sha(runtime/'app/native_policy_sensitivity.py'),
        design=dict(rows_per_period=256,periods=['TRAIN','INNER_A'],fits=0,optimizer_steps=0,test_access=False,
                    device='cpu',scope='hypothetical masks and diagnostic input perturbations, not portfolio scores',
                    fixed_rule_seeds=[7],cpu_gpu_bitwise_forward_claim=False))
    if (a.output/'IDENTITY.json').exists(): require(read(a.output/'IDENTITY.json')==identity,'AUDIT_IDENTITY')
    else: write(a.output/'IDENTITY.json',identity)
    local_spec=copy.deepcopy(spec)
    local_spec['engine_root']=str(a.engine_root)
    local_spec['inputs'].update(recipe=str(a.root/'RECIPE.json'),
        store=str(a.data_root/'full_pool_s_a1/full_pool_eurusd_h1.parquet'),
        manifest=str(a.data_root/'full_pool_s_a1/MANIFEST.json'),
        bars=str(a.data_root/'admitted/eurusd.parquet'),closures=str(a.data_root/'admitted/CLOSURE_INTERVALS.json'))
    full=SubsetMarket(local_spec,spec['dataset']['features'],'configured')
    require(support_hash(full.inputs)==spec['dataset']['support_hash'],'PAIRED_SUPPORT')
    proofs=[]
    for result_path in sorted((a.root/'tasks').glob('*-attempt-*/RESULT.json')):
        native=read(result_path);taskname=result_path.parent.name.split('-attempt-')[0]
        task=read(a.root/'tasks'/(taskname+'.json'))
        require(digest(task)==taskname,'TASK_IDENTITY')
        translated=copy.deepcopy(native)
        for artifact in translated['artifacts']:
            artifact['path']=str(local_artifact(artifact['path'],original_root,a.root))
        RLModelEvaluator().verify(translated,task['request'],task['params'],a.root)
        key=result_path.parent.name
        out=a.output/key;dest=out/'AUDIT.json'
        if dest.exists():
            proof=read(dest)
            require(proof['source_result_sha256']==file_sha(result_path)
                    and proof['audit_identity_sha256']==digest(identity),'CACHED_AUDIT_IDENTITY')
            for name,sha in proof['artifacts'].items():
                artifact=local_artifact(out/name,out,out)
                require(file_sha(artifact)==sha,'AUDIT_ARTIFACT')
            proofs.append(proof);continue
        write(a.output/'STATUS.json',dict(state='AUDITING',unit=key,completed=len(proofs),heartbeat_at=now()))
        model=PPO.load(result_path.parent/'model.zip',device='cpu')
        before=tensor_hash(model)
        training=read(result_path.parent/'TRAINING.json')
        require(before==training['final_weights'] and before!=training['initial_weights'],'WEIGHT_IDENTITY')
        market=object.__new__(SubsetMarket)
        market.base=full.base
        market.recipe=subset_recipe(full.base.recipe,native['features'])
        market.inputs=project_inputs(full.base.inputs,market.recipe)
        market.mode=native['cost_mode']
        require(read(result_path.parent/'RECIPE.json')==market.recipe,'CONSUMED_RECIPE')
        proof=dict(unit=key,source_result_sha256=file_sha(result_path),audit_identity_sha256=digest(identity),
            family=native['model_id'],cost_mode=native['cost_mode'],seed=native['seed'],features=len(native['features']),
            removed=[c for c in spec['dataset']['features'] if c not in native['features']],
            initial_weights=training['initial_weights'],final_weights=before,
            last_training_metrics=training['optimizer'],periods={},fixed_rules={},artifacts={})
        for period in ('TRAIN','INNER_A'):
            idx=np.flatnonzero(market.inputs.valid & (market.inputs.period==period))
            take=idx[np.unique(np.linspace(0,len(idx)-1,min(256,len(idx))).astype(int))]
            x=market.inputs.windows[take]
            result=probe_policy(model,x)
            result['sample_support_sha256']=hashlib.sha256(market.inputs.row_ts_ns[take].tobytes()).hexdigest()
            result['rows']=len(take)
            colmax=np.max(np.abs(x),axis=(0,1)); top=np.argsort(colmax)[-8:][::-1]
            result['sample_input_extremes']=[dict(feature=native['features'][j],max_abs=float(colmax[j])) for j in top]
            result['ledger']=ledger_checks(read(result_path.parent/(period+'.json')),
                                          read(result_path.parent/(period+'.trades.json')))
            proof['periods'][period]=result
        if task['cross'] and native['model_id']=='tcn' and native['seed']==7:
            rule='LONG_HOLD' if native['cost_mode']=='configured' else 'LONG_CLOSE'
            wrapper=FixedRule(model,rule)
            for period in ('TRAIN','INNER_A'):
                folder=out/rule
                evaluate(wrapper,market,period,folder)
                source=read(result_path.parent/(period+'.json')); replay=read(folder/(period+'.json'))
                fields=('net_profit_usd','sharpe_daily_sqrt365','max_drawdown_pct','trades')
                proof['fixed_rules'][period]=dict(rule=rule,
                    trades_exact=file_sha(folder/(period+'.trades.json'))==file_sha(result_path.parent/(period+'.trades.json')),
                    daily_equity_exact=file_sha(folder/(period+'.daily_equity.csv'))==file_sha(result_path.parent/(period+'.daily_equity.csv')),
                    metrics_exact=all(source[k]==replay[k] for k in fields),
                    metrics={k:replay[k] for k in fields})
        require(tensor_hash(model)==before,'AUDIT_CHANGED_WEIGHTS')
        proof['artifacts']={str(p.relative_to(out)):file_sha(p) for p in out.rglob('*') if p.is_file()}
        proof['finished_at']=now()
        write(dest,proof);proofs.append(proof)
        print(json.dumps(dict(unit=key,family=proof['family'],seed=proof['seed'],features=proof['features'],
            saturation=proof['periods']['INNER_A']['actor_tanh_saturated_fraction'],
            probability_span=proof['periods']['INNER_A']['routes']['flat']['probability_span_max'],
            fixed=proof['fixed_rules'])),flush=True)
        del model,market
    write(a.output/'SUMMARY.json',dict(identity=identity,units=len(proofs),reports=proofs,finished_at=now()))
    write(a.output/'STATUS.json',dict(state='SNAPSHOT_COMPLETE',units=len(proofs),fits=0,test_access=False,
        source_campaign_remains_live=True,finished_at=now()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','data-root','engine-root','selector-root'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
