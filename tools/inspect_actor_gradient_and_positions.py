"""CPU-only gradient/position diagnosis with frozen real TRAIN observations."""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from app.actor_learning_diagnostics import actor_gradient_report, matched_position_summary
from app.frozen_recipe_inputs import file_sha
from app.structural_refit import load_bundle
from app.policy_return_calibration import policy_signals
from app.three_signal_learning import emit
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import configure, read, write, now
from tools.run_structural_sac_refit import authenticate, market_for


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','source-runtime','degraded','anchor','calibration','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('NEW_GRADIENT_AUDIT_REQUIRED')
    configure(7)
    spec=authenticate(args.source,args.source_runtime)
    bundle=args.source/'runs/seed_7/bundles/epoch_0000'
    paths=dict(reference=bundle/'model.pt',degraded=args.degraded,anchor=args.anchor)
    hashes={k:file_sha(p) for k,p in paths.items()}
    if (hashes['degraded']!=read(args.degraded.parent/'MANIFEST.json')['files'][args.degraded.name]
            or hashes['anchor']!=read(args.anchor.parent/'DONE.json')['files'][args.anchor.name]):
        raise ValueError('GRADIENT_CHECKPOINT_HASH')
    verify_collection(args.calibration)
    if read(args.calibration/'CONFIG.json')['initial_sha256']!=hashes['reference']:
        raise ValueError('GRADIENT_CALIBRATION_LINEAGE')
    market=market_for(args.source,spec)
    ix=np.flatnonzero(market.inputs.valid & (market.inputs.period=='TRAIN'))
    rows=ix[np.linspace(0,len(ix)-1,1024,dtype=int)].reshape(8,128)
    gradients={}
    for label,path in paths.items():
        model,_=load_bundle(bundle,'cpu');model.load(path)
        before={n:p.detach().clone() for n,p in model.loss.named_parameters()}
        gradients[label]=[]
        for take in rows:
            batch=TensorDict(dict(features=torch.tensor(market.inputs.windows[take]),
                mask=torch.ones(len(take),3,dtype=torch.bool)),[len(take)])
            gradients[label].append(actor_gradient_report(model,batch))
        if not all(torch.equal(p,before[n]) for n,p in model.loss.named_parameters()):
            raise ValueError('GRADIENT_WEIGHTS_CHANGED')
        del model
    positions={}; reproduction={}
    for epoch in (0,40):
        model,_=load_bundle(bundle,'cpu')
        if epoch:model.load(args.degraded)
        with np.load(args.calibration/f'epoch_{epoch:04d}/SURFACES.npz') as archived:
            timestamps=archived['timestamp'].copy()
            archived_logits=archived['logits'].copy()
            archived_valid=archived['valid'].copy()
        if (not np.array_equal(timestamps,market.inputs.row_ts_ns)
                or not np.array_equal(archived_valid,market.inputs.valid)):
            raise ValueError('POSITION_SURFACE_SUPPORT')
        _, archive_probability=policy_signals(archived_logits, None)
        cpu_logits,_=emit(model,market.inputs)
        _, cpu_probability=policy_signals(cpu_logits, None)
        valid=market.inputs.valid
        # Reuse the established cross-device actor tolerance; probabilities used
        # for pairing below come exactly from the authenticated producing surfaces.
        if (not np.allclose(cpu_probability[valid],archive_probability[valid],atol=1e-5,rtol=1e-5)
                or not np.array_equal(cpu_logits[valid].argmax(-1),archived_logits[valid].argmax(-1))):
            raise ValueError(f'POSITION_CROSS_DEVICE_POLICY epoch={epoch} '
                f'max_error={np.max(np.abs(cpu_probability[valid]-archive_probability[valid]))} '
                f'actions={np.count_nonzero(cpu_logits[valid].argmax(-1)!=archived_logits[valid].argmax(-1))}')
        reproduction[str(epoch)]=dict(
            probability_max_abs_error=float(np.max(np.abs(cpu_probability[valid]-archive_probability[valid]))),
            greedy_action_disagreements=0,rtol=1e-5,atol=1e-5,archived_probabilities_exact=True)
        panels={}
        for seed in (7,19,43):
            directory=args.calibration/f'epoch_{epoch:04d}/stochastic_{seed}/TRAIN'
            with np.load(directory/'RETURN_TRACE.npz') as z:
                panel={k:z[k].copy() for k in ('timestamp','signal','position','reward','soft_return','action_probability')}
            trace=read(directory/'ACTIONS.json')
            for key in ('timestamp','signal','position','reward'):
                if not np.array_equal(panel[key],np.array([r[key] for r in trace])):
                    raise ValueError('POSITION_TRACE_MISMATCH')
            panel['command']=np.array([r['command'] for r in trace])
            loc=np.searchsorted(market.inputs.row_ts_ns,panel['timestamp'])
            if (not np.array_equal(market.inputs.row_ts_ns[loc],panel['timestamp'])
                    or not (market.inputs.period[loc]=='TRAIN').all() or not market.inputs.valid[loc].all()):
                raise ValueError('POSITION_OBSERVATION_SUPPORT')
            probabilities=archive_probability[loc,panel['signal']]
            if not np.array_equal(probabilities,panel['action_probability']):
                raise ValueError('POSITION_ARCHIVED_PROBABILITY')
            panels[seed]=panel
        positions[str(epoch)]={f'{a}_{b}':matched_position_summary(panels[a],panels[b])
            for a,b in combinations(panels,2)}
        del model
    if hashes!={k:file_sha(p) for k,p in paths.items()}:raise ValueError('GRADIENT_SOURCE_MUTATED')
    args.output.mkdir(parents=True)
    write(args.output/'REPORT.json',dict(source_hashes=hashes,source_identity=spec['identity'],
        tool_sha256=file_sha(Path(__file__)),
        helper_sha256=file_sha(Path(__file__).resolve().parents[1]/'app/actor_learning_diagnostics.py'),
        gradients=gradients,positions=positions,policy_reproduction=reproduction,rows=1024,batches=8,seed=7,
        optimizer_steps=0,source_unchanged=True,fit_period='TRAIN',model_promoted=False))
    write(args.output/'DONE.json',dict(report_sha256=file_sha(args.output/'REPORT.json'),finished_at=now()))
    print(dict(snapshots=len(gradients),gradient_batches=24,position_pairs=6,source_unchanged=True))


if __name__=='__main__':main()
