"""Sequential deletion and full phase2 frozen donor-assisted readmission."""
import argparse
import copy
import fcntl
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from agent_plugins.three_signal_sac import ThreeLogits
from app.cumulative_readmission import cumulative_pass, expand_actor
from app.frozen_actor_compaction import effective_actor, compare_logits, PackedActor
from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import cached_measurement, masked_emit, prepared_signals
from app.recipe_runtime import RecipeMarket
from app.rl_subset import coherent_blocks, digest, project_inputs, subset_recipe, support_hash
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_refit_mask_pass import account_equal, score_metrics

ROOT=Path(__file__).resolve().parents[1]


def prepare(args):
    if args.root.exists(): raise ValueError('NEW_CUMULATIVE_ROOT')
    parent=read(args.parent/'SPEC.json'); verify_files(args.parent,parent['files'])
    base=read(args.baseline/'MANIFEST.json'); verify_files(args.baseline,base['files'])
    ancestor=args.parent/'SOURCE_BUNDLE'
    verify_files(ancestor,read(ancestor/'MANIFEST.json')['files'])
    if file_sha(args.archive/'RESULT.json')!=args.archive_receipt_sha:
        raise ValueError('ARCHIVE_RECEIPT')
    receipt=read(args.archive/'RESULT.json')
    for name in ('RECIPE.json','SOURCE_RECEIPT.json'):
        if file_sha(args.archive/name)!=receipt['artifacts'][name]:
            raise ValueError('ARCHIVE_PROVENANCE:'+name)
    old=args.archive/'checkpoints'/f'{args.archive_step}.pt'
    if receipt['artifacts'][f'checkpoints/{args.archive_step}.pt']!=file_sha(old):
        raise ValueError('ARCHIVE_CHECKPOINT_RECEIPT')
    universe=read(args.parent/'SUPPORT_RECIPE.json')['list']['columns']
    original=read(args.original_selection/'SPEC.private.json')
    if universe!=original['dataset']['features']: raise ValueError('ORIGINAL_PHASE2_UNIVERSE')
    active=read(args.baseline/'RECIPE.json')['list']['columns']
    originals=[args.parent/'SPEC.json',args.baseline.parent/'RESULT.json',
        args.original_selection/'SPEC.private.json',args.archive/'RECIPE.json',
        args.archive/'SOURCE_RECEIPT.json',args.archive/'RESULT.json',old]
    originals.extend(args.parent/p for p in parent['files'])
    originals.extend(p for p in args.baseline.iterdir() if p.is_file())
    sources=set(parent['sources'])|{'app/cumulative_readmission.py','tools/run_cumulative_mask_pass.py',
        'app/frozen_actor_compaction.py'}
    spec=dict(schema='cumulative_phase2.v1',created_at=now(),parent=str(args.parent),
        baseline=str(args.baseline),ancestor=str(ancestor),archive=str(args.archive),
        archive_step=args.archive_step,original_selection=str(args.original_selection),
        source_files={str(p):file_sha(p) for p in originals},
        runtime_files={p:file_sha(ROOT/p) for p in sorted(sources)},
        initial=active,universe=universe,blocks=coherent_blocks(universe),
        device=args.device,ranking='VALIDATION return/week; >2 orders/week both periods',
        mode='DONOR_ASSISTED_FROZEN_READMISSION',optimizer_steps=0,test_read=False)
    spec['identity']=digest(spec);args.root.mkdir(parents=True)
    write(args.root/'SPEC.private.json',spec)
    print(spec['identity'],flush=True)


def authenticate(root):
    s=read(root/'SPEC.private.json')
    if s['identity']!=digest({k:v for k,v in s.items() if k!='identity'}): raise ValueError('SPEC_CHANGED')
    for p,h in s['source_files'].items():
        if file_sha(p)!=h: raise ValueError('SOURCE_CHANGED:'+p)
    verify_files(ROOT,s['runtime_files'])
    return s


def make_expanded(s):
    base,_=load_bundle(Path(s['baseline']),s['device'])
    ancestor,_=load_bundle(Path(s['ancestor']),s['device'])
    oldrec=read(Path(s['archive'])/'RECIPE.json')
    old=ThreeLogits(oldrec).to(s['device']).eval()
    state=torch.load(Path(s['archive'])/'checkpoints'/f"{s['archive_step']}.pt",
                     map_location=s['device'],weights_only=False)['loss']
    prefix='actor_network_params.module.0.module.'
    old.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
    if any(not torch.isfinite(v).all() for v in old.state_dict().values()): raise ValueError('DONOR_NONFINITE')
    universe=read(Path(s['parent'])/'SUPPORT_RECIPE.json')
    donors=[(effective_actor(ancestor),ancestor.recipe,'ancestor76'),(old,oldrec,'original120')]
    expanded,restoration=expand_actor(effective_actor(base),base.recipe,universe,donors)
    config=dict(base_recipe=base.recipe,base_architecture=base.effective['architecture'],
        universe=universe,donors=[dict(recipe=r,label=label,
            architecture=ancestor.effective['architecture'] if label=='ancestor76' else {}) for _,r,label in donors])
    return base,expanded,restoration,config


def load_selected(path,device='cpu'):
    manifest=read(path/'MANIFEST.json');verify_files(path,manifest['files'])
    cfg=read(path/'CONFIG.json')
    base=ThreeLogits(cfg['base_recipe'],cfg['base_architecture']).to(device).eval()
    donors=[(ThreeLogits(d['recipe'],d['architecture']).to(device).eval(),d['recipe'],d['label']) for d in cfg['donors']]
    actor,_=expand_actor(base,cfg['base_recipe'],cfg['universe'],donors)
    actor.load_state_dict(torch.load(path/'actor.pt',map_location=device,weights_only=True),strict=True)
    names=cfg['universe']['list']['columns']
    packed=PackedActor(actor,[names.index(c) for c in cfg['selected']],len(names)).to(device)
    return SimpleNamespace(device=torch.device(device),surfaces=packed.surface),cfg


def run(args):
    root=args.root;s=authenticate(root)
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'DONE.json').exists():
            verify_files(root,read(root/'DONE.json')['files']);return
        configure(7);torch.backends.cudnn.allow_tf32=False
        if s['device']=='cuda' and 'RTX 5090' not in torch.cuda.get_device_name():
            raise ValueError('INTENDED_GPU_REQUIRED')
        parent=Path(s['parent']);ps=read(parent/'SPEC.json')
        paths={k:str(parent/v) for k,v in ps['inputs'].items()}
        paths['recipe']=str(parent/'SUPPORT_RECIPE.json')
        market=RecipeMarket(dict(inputs=paths,engine_root=str(parent/'engine'),
            recipe_sha256=file_sha(paths['recipe'])))
        if support_hash(market.inputs)!=read(parent/'PREFLIGHT.json')['support_sha256']:
            raise ValueError('CUMULATIVE_SUPPORT')
        base,actor,restoration,cfg=make_expanded(s)
        model=SimpleNamespace(device=torch.device(s['device']),surfaces=actor.surface)
        universe=s['universe'];risk=read(parent/'RISK.json');d=ps['design']
        start=time.monotonic();times=[]
        def assess(cols,save=None,diagnose=False):
            logits=masked_emit(model,market.inputs,[c in cols for c in universe])
            prepared=prepared_signals(logits)
            metrics={p:evaluate(environment(market,risk,d['horizon'],interval,ps['effective']['gamma']),
                logits,prepared=prepared,diagnose=diagnose,save=save/p if save else None)
                for p,interval in market.recipe['split']['periods'].items()}
            return score_metrics(metrics),logits
        control,logits=assess(s['initial'],diagnose=True)
        initial_inputs=project_inputs(market.inputs,base.recipe)
        reference,_=emit(base,initial_inputs)
        parity=compare_logits(reference[market.inputs.valid],logits[market.inputs.valid])
        if not parity['qualified']: raise ValueError('CUMULATIVE_ACTOR_PARITY')
        expected=read(Path(s['baseline']).parent/'RESULT.json')['metrics']
        for p,m in control['metrics'].items():account_equal(m,expected[p])
        write(root/'PREFLIGHT.json',dict(identity=s['identity'],parity=parity,
            accounts_exact=True,rows=int(market.inputs.valid.sum()),support_hash=support_hash(market.inputs),
            restored=restoration,control=control,device_name=torch.cuda.get_device_name() if s['device']=='cuda' else 'cpu'))
        write(root/'INVENTORY.json',dict(universe=universe,initial=s['initial'],
            historical_phase2_absent=[c for c in universe if c not in s['initial']],restoration=restoration))
        if args.preflight_only:return
        context=dict(identity=s['identity'],device=s['device'])
        def measure(cols):
            def fresh():
                begin=time.monotonic();value=control if cols==s['initial'] else assess(cols)[0]
                times.append(time.monotonic()-begin)
                write(root/'PROGRESS.json',dict(at=now(),features=len(cols),fresh_evaluations=len(times),
                    last_seconds=times[-1],median_seconds=float(np.median(times)),seconds=time.monotonic()-start))
                return value
            return cached_measurement(root/'cache',context,cols,fresh)
        result=cumulative_pass(universe,s['blocks'],s['initial'],measure,
            lambda r:write(root/'PASSES.json',dict(r,at=now())))
        selected=result['selected'];final,_=assess(selected,save=root/'final',diagnose=True)
        if final['score']<control['score'] or not final['eligible']:raise ValueError('FINAL_REGRESSION')
        for p,m in final['metrics'].items():account_equal(m,measure(selected)['metrics'][p])
        bundle=root/'bundle';bundle.mkdir(exist_ok=True)
        cfg.update(selected=selected,restoration=restoration,selection_identity=s['identity'],
            input_recipe=subset_recipe(cfg['universe'],selected),
            scope='frozen packed donor-restored actor; not fresh SAC training or live promotion')
        write(bundle/'CONFIG.json',cfg);torch.save(actor.state_dict(),bundle/'actor.pt')
        write(bundle/'MANIFEST.json',dict(schema='cumulative_frozen_actor.v1',
            files={n:file_sha(bundle/n) for n in ('CONFIG.json','actor.pt')}))
        reload_model,_=load_selected(bundle,s['device'])
        packed_inputs=project_inputs(market.inputs,cfg['input_recipe'])
        actual,_=emit(reload_model,packed_inputs)
        wanted=masked_emit(model,market.inputs,[c in selected for c in universe])
        if not np.array_equal(actual[market.inputs.valid],wanted[market.inputs.valid]):
            raise ValueError('CUMULATIVE_RELOAD')
        prepared=prepared_signals(actual)
        for p,interval in market.recipe['split']['periods'].items():
            m=evaluate(environment(market,risk,d['horizon'],interval,ps['effective']['gamma']),actual,prepared=prepared)
            account_equal(m,final['metrics'][p])
        restored,_=assess(s['initial'])
        for p,m in restored['metrics'].items():account_equal(m,control['metrics'][p])
        authenticate(root)
        write(root/'RESULT.json',dict(identity=s['identity'],result=result,control=control,final=final,
            baseline_parity=parity,source_unchanged=True,reload_exact=True,reload_accounts_exact=True,
            mode=s['mode'],historical_pool_complete=True,optimizer_steps=0,test_read=False,
            seconds=time.monotonic()-start,model_promoted=False))
        write(root/'DONE.json',dict(identity=s['identity'],finished_at=now(),files={
            str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()
            and p.name not in ('DONE.json','run.lock','PROGRESS.json') and p.suffix!='.lock'}))
        print('COMPLETE',result['counts'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare')
    for key in ('root','parent','baseline','archive','original-selection'):a.add_argument('--'+key,type=Path,required=True)
    a.add_argument('--archive-step',type=int,required=True);a.add_argument('--archive-receipt-sha',required=True)
    a.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=sub.add_parser('run');a.add_argument('--root',type=Path,required=True);a.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();globals()[args.command](args)


if __name__=='__main__':main()
