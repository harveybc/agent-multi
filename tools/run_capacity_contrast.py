"""Prepare a width-only SAC capacity contrast against an authenticated control."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from tools.run_generalization_queue import ROOT, read, write, sha


def widened(architecture, width):
    expected = {'branch_width', 'core_width', 'head_width', 'branch_blocks',
                'core_blocks', 'temporal_attention'}
    if set(architecture) != expected or type(width) is not int or width < 8 or width % 2:
        raise ValueError('CAPACITY_ARCHITECTURE')
    if (architecture['branch_width'], architecture['core_width'], architecture['head_width']) != (8, 16, 8):
        raise ValueError('CAPACITY_BASE_WIDTHS')
    attention = architecture['temporal_attention']
    if (attention['kind'], attention['embedding'], attention['feedforward'], attention['heads']) != ('causal_patch', 8, 16, 2):
        raise ValueError('CAPACITY_BASE_ATTENTION')
    out = copy.deepcopy(architecture)
    out.update(branch_width=width, core_width=2*width, head_width=width)
    out['temporal_attention'].update(embedding=width, feedforward=2*width)
    return out


def meta_count(recipe, architecture):
    import torch
    from agent_plugins.three_signal_sac import ThreeLogits
    with torch.device('meta'):
        network = ThreeLogits(recipe, architecture)
    actor = sum(p.numel() for p in network.parameters() if p.requires_grad)
    return dict(actor=actor, online_twin_critics=2*actor, temperature=1,
                trainable=3*actor+1, target_parameter_copies=2*actor)


def choose_width(recipe, architecture):
    base = meta_count(recipe, architecture)
    for width in range(8, 514, 2):
        candidate = widened(architecture, width)
        count = meta_count(recipe, candidate)
        ratio = count['trainable']/base['trainable']
        if ratio >= 70:
            if ratio > 75:
                raise ValueError('CAPACITY_RATIO_OVERSHOOT')
            return candidate, dict(base=base, candidate=count, ratio=ratio, width=width,
                capacity_definition='unique trainable actor + online twin critics + temperature',
                selection='smallest common even width reaching70x; no performance search')
    raise ValueError('CAPACITY_WIDTH_NOT_FOUND')


def actual_count(model):
    trainable = {id(p): p for p in model.loss.parameters() if p.requires_grad}
    optimized = [p for group in model.optimizer.param_groups for p in group['params']]
    if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != set(trainable):
        raise ValueError('CAPACITY_OPTIMIZER_COVERAGE')
    return sum(p.numel() for p in trainable.values())


def prepare(args):
    from app.structural_refit import build_model
    from app.rl_subset import digest, support_hash
    from tools.run_structural_sac_refit import authenticate, market_for, SOURCES
    source = authenticate(args.source_root, args.source_runtime)
    if source['structural_round']['variant'] != 'R0' or source['effective'].get('critic_regularization'):
        raise ValueError('CAPACITY_REQUIRES_UNREGULARIZED_ATTENTION_CONTROL')
    recipe = read(args.source_root/'RECIPE.json')
    architecture, audit = choose_width(recipe, source['effective']['architecture'])
    effective = copy.deepcopy(source['effective'])
    effective['architecture'] = architecture
    model = build_model(recipe, effective)
    audit['actual_trainable'] = actual_count(model)
    if audit['actual_trainable'] != audit['candidate']['trainable'] or model.effective != effective:
        raise ValueError('CAPACITY_EFFECTIVE_MISMATCH')
    del model
    if args.root.exists():
        raise ValueError('NEW_ROOT_REQUIRED')
    args.root.mkdir(parents=True)
    for name, expected in source['files'].items():
        target = args.root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source_root/name, target)
        if sha(target) != expected:
            raise ValueError('CAPACITY_COPY_CHANGED:'+name)
    spec = copy.deepcopy(source)
    spec.pop('identity')
    spec['created_at'] = datetime.now(timezone.utc).isoformat()
    spec['effective'] = effective
    spec['design']['seeds'] = [7, 19, 31]
    spec['structural_round'] = dict(variant='CAPACITY_70X', parent_identity=source['identity'],
        control_identity=source['structural_round']['control_identity'],
        fresh_weights=True, fresh_replay=True, context_transport='original13x93',
        only_intervention='uniform width increase; layers and receptive field unchanged',
        pilot_hardware='primary matches control; fallback comparison cross-device; replications device-matched')
    write(args.root/'CAPACITY.json', audit)
    spec['files']['CAPACITY.json'] = sha(args.root/'CAPACITY.json')
    spec['sources'] = {name: sha(ROOT/name) for name in SOURCES +
        ['tools/run_generalization_queue.py', 'tools/run_capacity_contrast.py']}
    spec['identity'] = digest(spec)
    write(args.root/'SPEC.json', spec)
    market = market_for(args.root, spec)
    preflight = read(args.source_root/'PREFLIGHT.json')
    if support_hash(market.inputs) != preflight['support_sha256']:
        raise ValueError('CAPACITY_SUPPORT_CHANGED')
    preflight.update(identity=spec['identity'], structural_round=spec['structural_round'],
        capacity=audit, fresh_training=True, tensor_shape=list(market.inputs.windows.shape))
    write(args.root/'PREFLIGHT.json', preflight)
    print(dict(identity=spec['identity'], capacity=audit), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-runtime', 'root'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
