"""Frozen-policy greedy versus categorical sampling on unchanged native accounts."""
import argparse
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle
from app.three_signal_learning import emit, evaluate, environment
from app.training_forensics import tree_digest
from tools.analyze_online_transients import compact
from tools.run_discrete_curve_probe import configure, read, write
from tools.run_long_sac_continuation import SUPPORT, checked_baseline


def sampled_signals(logits, uniforms):
    if logits.ndim != 2 or logits.shape[1] != 3 or uniforms.shape != (len(logits),):
        raise ValueError('SAMPLING_SHAPE')
    if not np.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms >= 1).any():
        raise ValueError('SAMPLING_UNIFORMS')
    prepared = list(prepared_signals(logits))
    valid = prepared[3]
    raw = logits[valid].astype(float)
    prob = np.exp(raw-raw.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    # The final category catches numerical CDF roundoff without clipping inputs.
    prepared[0][valid] = (uniforms[valid, None] >= np.cumsum(prob, axis=1)[:, :2]).sum(axis=1)
    return tuple(prepared)


def run(a):
    if torch.cuda.is_available():
        raise ValueError('FORCE_CPU')
    configure(7)
    if a.output.exists():
        raise ValueError('NEW_OUTPUT_REQUIRED')
    a.output.mkdir(parents=True)
    parent = read(a.parent/'SPEC.json'); risk = read(a.parent/'RISK.json')
    rows = []
    for seed in (7, 19, 43):
        root = a.snapshots/f'seed_{seed}'
        history = read(root/'HISTORY.json'); design = read(root/'DESIGN.json')
        model, _ = load_bundle(root/'initial_bundle', 'cpu')
        market = market_for(a.parent, parent, model.recipe)
        if support_hash(market.inputs) != SUPPORT:
            raise ValueError('SAMPLING_SUPPORT')
        for epoch in ([0, len(history)] if seed == 7 else [len(history)]):
            if epoch:
                path = root/'weights'/f'epoch_{epoch:04d}.pt'
                if file_sha(path) != history[-1]['weight_snapshot']['sha256']:
                    raise ValueError('SAMPLING_WEIGHT_HASH')
                model.loss.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            before = tree_digest(model.loss.state_dict())
            logits, _ = emit(model, market.inputs)
            def account(prepared):
                return {p: evaluate(environment(market, risk, design['horizon'], interval, model.effective['gamma']),
                                    logits, prepared=prepared)
                        for p, interval in model.recipe['split']['periods'].items()}
            greedy = account(prepared_signals(logits))
            if epoch == 0:
                checked_baseline(greedy)
            stochastic = []
            for draw_seed in (101, 202, 303):
                uniforms = np.random.default_rng(draw_seed).random(len(logits))
                sampled = account(sampled_signals(logits, uniforms))
                stochastic.append(dict(draw_seed=draw_seed, metrics=compact(sampled)))
            if before != tree_digest(model.loss.state_dict()):
                raise ValueError('SAMPLING_MUTATED_WEIGHTS')
            rows.append(dict(seed=seed, epoch=epoch, greedy=compact(greedy), sampled=stochastic))
            write(a.output/'RESULT.json', dict(rows=rows, complete=False, trained=False, source_support=SUPPORT))
            print(f'frozen_sampling seed={seed} epoch={epoch} complete', flush=True)
    write(a.output/'RESULT.json', dict(rows=rows, complete=True, trained=False, source_support=SUPPORT,
        scope='Fixed policies: distributional comparison, not replay of the changing-policy training account',
        source_sha256=file_sha(Path(__file__))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshots', 'parent', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
