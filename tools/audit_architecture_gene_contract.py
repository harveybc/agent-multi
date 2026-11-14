"""CPU construction witnesses for post-phase2 genes; no training or DEAP fitness."""
import argparse
import copy
import math
from pathlib import Path

import numpy as np
import torch

from app.architecture_gene_contract import append_event, decode, encode, read_ledger, value_at
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import build_model, verify_files
from tools.run_discrete_curve_probe import read, write


def observe(model):
    a, q = model.actor_net, model.q_net
    e = a.encoder
    attn = e.post_fusion
    return {
        'architecture.branch_width': [b[0].conv.out_channels for b in e.branches],
        'architecture.branch_blocks': [len(b)-1 for b in e.branches],
        'architecture.core_width': e.fusion.conv.out_channels,
        'architecture.core_blocks': len(e.core),
        'architecture.head_width': q.head[0].out_features,
        'architecture.actor_head_width': a.head[0].out_features,
        'architecture.temporal_attention.patch_len': attn.patch_projection.in_features,
        'architecture.temporal_attention.embedding': attn.patch_projection.out_features,
        'architecture.temporal_attention.heads': attn.encoder.self_attn.num_heads,
        'architecture.temporal_attention.feedforward': attn.encoder.linear1.out_features,
        'architecture.temporal_attention.gate_init': float(attn.gate.detach()),
        'learning_rate': [g['lr'] for g in model.optimizer.param_groups],
        'gamma': float(model.loss.value_estimator.gamma),
        'tau': 1-model.updater.eps,
        'weight_decay': [g['weight_decay'] for g in model.optimizer.param_groups],
        'entropy_fraction': model.entropy_fraction,
        'alpha': float(model.loss.log_alpha.detach().exp()),
        'critic_regularization.p': [b.dropout1.p for b in q.encoder.core] +
                                   [b.dropout2.p for b in q.encoder.core],
    }


def assert_observed(config, observed):
    for path, values in observed.items():
        expected = value_at(config, path)
        values = values if isinstance(values, list) else [values]
        if not values or any(not math.isclose(v, expected, rel_tol=1e-6, abs_tol=1e-8) for v in values):
            raise ValueError('GENE_NOT_CONSUMED:' + path)


def run(a):
    if a.output.exists():
        raise ValueError('NEW_GENE_AUDIT_ROOT')
    verify_files(a.bundle, read(a.bundle/'MANIFEST.json')['files'])
    closure = read(a.closure)
    if not (closure['ready_for_architecture_search'] and closure['cpu_accounts_exact'] == 2
            and closure['status'] in ('LOCAL_FROZEN_NEIGHBORHOOD_STABLE', 'BUDGET_CLOSED_NOT_LOCAL_OPTIMUM')):
        raise ValueError('GENE_PHASE2_EXIT')
    recipe = read(a.bundle/'RECIPE.json')
    if recipe['list']['columns'] != closure['selected']:
        raise ValueError('GENE_SELECTED_INPUTS')
    space = read(a.space)
    base = read(a.bundle/'MODEL_CONFIG.json')
    base = copy.deepcopy(base)
    base['architecture']['actor_head_width'] = base['architecture']['head_width']
    genome = encode(space, base)
    identity = digest(dict(space=space, effective=base, recipe=file_sha(a.bundle/'RECIPE.json'),
                           closure=file_sha(a.closure),
                           source_files={str(p.relative_to(Path(__file__).resolve().parents[1])):file_sha(p)
                            for folder in ('app','agent_plugins','tools')
                            for p in (Path(__file__).resolve().parents[1]/folder).rglob('*.py')}))
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(a.bundle/'PROBE.npz') as saved_probe:
        probe = torch.tensor(saved_probe['inputs'][:3])
    cases = [('baseline', genome)]
    for i, gene in enumerate(space['genes']):
        points = [0., 1.]
        if gene['kind'] == 'choice':
            points += [(j+.5)/len(gene['values']) for j in range(len(gene['values']))]
        else:
            points += [.5]
        for j, x in enumerate(points):
            raw = list(genome)
            raw[i] = x
            cases.append((gene['path']+'/'+str(j), raw))
    receipts, phenotypes = [], {}
    for index, (label, raw) in enumerate(cases):
        cfg = decode(space, raw, base)
        key = digest(cfg)
        event = dict(campaign_id=identity, candidate_id=f'construction_{index}',
                     stage='CONSTRUCTION_ONLY_NOT_FITNESS', status='REQUESTED',
                     raw_genome=raw, decoded_config=cfg, seed=7, parents=[],
                     metrics={}, artifacts={}, label=label, phenotype_id=key)
        append_event(a.output/'CANDIDATES.jsonl', event)
        if key in phenotypes:
            append_event(a.output/'CANDIDATES.jsonl', dict(event, status='CACHE_HIT',
                         reused_candidate=phenotypes[key]))
            continue
        try:
            model = build_model(recipe, cfg, seed=7, device='cpu')
            actual = observe(model)
            assert_observed(cfg, actual)
            with torch.no_grad():
                z = model.surfaces(probe).clone()
            if z.shape != (len(probe), 3) or not torch.isfinite(z).all():
                raise ValueError('GENE_OUTPUT')
            folder = a.output/key
            folder.mkdir()
            model.save(folder/'model.pt')
            cold = build_model(recipe, cfg, seed=7, device='cpu')
            cold.load(folder/'model.pt')
            with torch.no_grad():
                if not torch.equal(z, cold.surfaces(probe)):
                    raise ValueError('GENE_RELOAD')
            assert_observed(cfg, observe(cold))
            stats = {name:dict(parameters=sum(p.numel() for p in module.parameters()),
                              parameter_l2=float(sum(p.detach().double().square().sum()
                                                     for p in module.parameters()).sqrt()))
                     for name, module in [('actor_branches',model.actor_net.encoder.branches),
                         ('actor_fusion',model.actor_net.encoder.fusion),
                         ('actor_attention',model.actor_net.encoder.post_fusion),
                         ('actor_core',model.actor_net.encoder.core),('actor_head',model.actor_net.head)]}
            receipt = dict(candidate_id=event['candidate_id'], label=label, phenotype_id=key,
                           requested=cfg, effective=model.effective, actual=actual,
                           components=stats, output_shape=list(z.shape), cold_logits_exact=True,
                           positional_shape=list(model.actor_net.encoder.post_fusion.positions.shape),
                           test_read=False, optimizer_steps=0, trading_metrics=None)
            write(folder/'RECEIPT.json', receipt)
            artifacts={str(p.relative_to(a.output)):file_sha(p) for p in folder.iterdir()}
            append_event(a.output/'CANDIDATES.jsonl', dict(event, status='COMPLETE',
                         actual_parameters=actual, metrics={'components':stats}, artifacts=artifacts))
            phenotypes[key] = event['candidate_id']
            receipts.append(receipt)
            del model, cold
        except Exception as exc:
            append_event(a.output/'CANDIDATES.jsonl', dict(event, status='FAILED',
                         error_type=type(exc).__name__, error=str(exc)))
            raise
    ledger = read_ledger(a.output/'CANDIDATES.jsonl')
    result = dict(identity=identity, genes=len(space['genes']), requests=len(cases),
                  constructed=len(receipts), cached_aliases=len(cases)-len(receipts),
                  ledger_events=len(ledger), ledger_tail=ledger[-1]['sha256'],
                  cold_reload_exact=len(receipts), receipts=receipts,
                  limitations=space['not_yet_enabled'], training_path_verified=False,
                  entropy_fraction_scope='constructor field only; actual update-path witness still required',
                  source_unchanged=True, deap_launched=False, optimizer_steps=0)
    verify_files(a.bundle, read(a.bundle/'MANIFEST.json')['files'])
    write(a.output/'RESULT.json', result)
    print('CONSTRUCTION_VERIFIED',len(receipts),'phenotypes',len(cases),'requests',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','closure','space','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
