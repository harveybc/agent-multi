"""Copied365 CPU probes, native replays and component-local restoration diagnostics."""
import argparse
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.frozen_signal_masks import prepared_signals
from app.online_transient_forensics import checkpoint_drift, functional_metrics
from app.rollback_activity import actor_pressure, swap_component
from app.rl_subset import support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from app.training_forensics import tree_digest, rng_state
from tools.diagnose_sac_collection_policy import sampled_signals
from tools.run_discrete_curve_probe import configure, read, write
from tools.run_long_sac_continuation import SUPPORT
from tools.run_structural_sac_refit import assess_full


PERIODS = ('TRAIN', 'VALIDATION')


def probes(model, windows, baseline):
    result = {}
    with torch.no_grad():
        for period, x in windows.items():
            z, twins = model.surfaces(x), model.q_surfaces(x)
            alpha = float(model.loss.log_alpha.detach().exp())
            row = functional_metrics(z, twins, baseline[period], alpha)
            q = twins.min(0).values
            grad = actor_pressure(z, q, alpha)
            row.update(mean_probability=z.softmax(-1).mean(0).tolist(),
                q_greedy_share=[float((q.argmax(-1) == i).float().mean()) for i in range(3)],
                q_advantage_over_nop=(q - q[:, :1]).mean(0).tolist(),
                mean_logit_gradient=grad.mean(0).tolist(),
                nop_upward_pressure_share=float((grad[:, 0] < 0).double().mean()))
            result[period] = row
    return result


def parity(actual, expected):
    rows = {p: dict(return_delta=actual[p]['return_pct'] - expected[p]['return_pct'],
                    trades_equal=actual[p]['trades'] == expected[p]['trades'],
                    signals_equal=actual[p]['signals'] == expected[p]['signals']) for p in PERIODS}
    return dict(periods=rows, passed=all(abs(v['return_delta']) < 1e-8
        and v['trades_equal'] and v['signals_equal'] for v in rows.values()))


def run(a):
    if torch.cuda.is_available():
        raise ValueError('FORCE_CPU')
    configure(7)
    a.output.mkdir(parents=True, exist_ok=False)
    parent = read(a.parent / 'SPEC.json')
    verify_files(a.parent, {k: v for k, v in parent['files'].items()
        if k.startswith(('data/', 'engine/')) or k in ('SUPPORT_RECIPE.json', 'RISK.json')})
    risk = read(a.parent / 'RISK.json')
    results = []
    for arm in ('A', 'D', 'control'):
        root = a.snapshots / arm
        collection = read(root / 'COLLECTION.json')
        verify_files(root, collection['files'])
        history, design = read(root / 'HISTORY.json'), read(root / 'DESIGN.json')
        model, _ = load_bundle(root / 'initial_bundle', 'cpu')
        initial = {k: v.clone() for k, v in model.loss.state_dict().items()}
        market = market_for(a.parent, parent, model.recipe)
        if support_hash(market.inputs) != SUPPORT or set(model.recipe['split']['periods']) != set(PERIODS):
            raise ValueError('SUPPORT_OR_SPLITS_CHANGED')
        windows, baseline = {}, {}
        for period, count in (('TRAIN', 512), ('VALIDATION', 256)):
            valid = np.flatnonzero(market.inputs.valid & (market.inputs.period == period))
            ix = valid[np.linspace(0, len(valid) - 1, count, dtype=int)]
            windows[period] = torch.as_tensor(market.inputs.windows[ix].copy())
            with torch.no_grad():
                baseline[period] = model.surfaces(windows[period]).clone()
        states = {0: initial}
        for path in sorted((root / 'weights').glob('epoch_*.pt')):
            epoch = int(path.stem.split('_')[-1])
            row = next(h for h in history if h['epoch'] == epoch)
            if file_sha(path) != row['weight_snapshot']['sha256']:
                raise ValueError('SNAPSHOT_HASH')
            states[epoch] = torch.load(path, map_location='cpu', weights_only=True)
        events = [h['best_rollback'] for h in history if h.get('best_rollback')]
        best_epoch = events[-1]['restored_epoch']
        best, latest = states[best_epoch], max(states)
        for event in events:
            if tree_digest(states[event['restored_epoch']]) != event['after_weights_sha256']:
                raise ValueError('ROLLBACK_TARGET_DIGEST')
        receipt = dict(arm=arm, best_epoch=best_epoch, latest_epoch=latest, points=[],
            swaps=[], sampling=[], rollback_receipts=events, rollback_target_digests_exact=True,
            collection_sha256=file_sha(root / 'COLLECTION.json'))
        for epoch, state in sorted(states.items()):
            model.loss.load_state_dict(state)
            before, rng = tree_digest(model.loss.state_dict()), tree_digest(rng_state())
            metrics = assess_full(model, market, risk, design)
            expected = read(root / 'EPOCH_ZERO.json')['metrics'] if not epoch else next(
                h['metrics'] for h in history if h['epoch'] == epoch)
            receipt['points'].append(dict(epoch=epoch, metrics=metrics,
                native_parity=parity(metrics, expected), probe=probes(model, windows, baseline),
                drift_vs_initial=checkpoint_drift(state, initial), drift_vs_best=checkpoint_drift(state, best)))
            if before != tree_digest(model.loss.state_dict()) or rng != tree_digest(rng_state()):
                raise ValueError('PROBE_MUTATED_STATE_OR_RNG')
            write(a.output / (arm + '.json'), receipt)
            print(f'{arm} epoch={epoch} replay_parity={receipt["points"][-1]["native_parity"]["passed"]}', flush=True)
        current = states[latest]
        best_metrics = next(p['metrics'] for p in receipt['points'] if p['epoch'] == best_epoch)
        latest_metrics = receipt['points'][-1]['metrics']
        for part in ('unchanged', 'branches', 'fusion', 'attention', 'core', 'head', 'all'):
            state, keys = (current, []) if part == 'unchanged' else swap_component(current, best, part)
            model.loss.load_state_dict(state)
            before, rng = tree_digest(model.loss.state_dict()), tree_digest(rng_state())
            metrics = assess_full(model, market, risk, design)
            if part in ('unchanged', 'all') and not parity(metrics, latest_metrics if part == 'unchanged' else best_metrics)['passed']:
                raise ValueError('SWAP_IDENTITY_CONTROL_FAILED')
            receipt['swaps'].append(dict(restored_component=part, keys=keys, metrics=metrics,
                probe=probes(model, windows, baseline)))
            if before != tree_digest(model.loss.state_dict()) or rng != tree_digest(rng_state()):
                raise ValueError('SWAP_MUTATED_STATE_OR_RNG')
            write(a.output / (arm + '.json'), receipt)
            print(f'{arm} swap={part} complete', flush=True)
        for epoch in sorted({best_epoch, latest}):
            model.loss.load_state_dict(states[epoch])
            before, rng = tree_digest(model.loss.state_dict()), tree_digest(rng_state())
            logits, _ = emit(model, market.inputs)
            def account(prepared):
                return {p: evaluate(environment(market, risk, design['horizon'], interval,
                    model.effective['gamma']), logits, prepared=prepared)
                    for p, interval in model.recipe['split']['periods'].items()}
            greedy = account(prepared_signals(logits))
            sampled = [dict(seed=seed, metrics=account(sampled_signals(logits,
                np.random.default_rng(seed).random(len(logits))))) for seed in (101, 202, 303)]
            receipt['sampling'].append(dict(epoch=epoch, greedy=greedy, sampled=sampled))
            if before != tree_digest(model.loss.state_dict()) or rng != tree_digest(rng_state()):
                raise ValueError('SAMPLING_MUTATED_STATE_OR_RNG')
        model.loss.load_state_dict(initial)
        if tree_digest(model.loss.state_dict()) != tree_digest(initial):
            raise ValueError('FINAL_RESTORATION_FAILED')
        verify_files(root, collection['files'])
        write(a.output / (arm + '.json'), receipt)
        results.append(receipt)
    write(a.output / 'RESULT.json', dict(complete=True, results=results, trained=False,
        promoted=False, live_workers_modified=False, support_sha256=SUPPORT,
        all_native_parity_pass=all(p['native_parity']['passed'] for r in results for p in r['points']),
        analysis_sources={name: file_sha(Path(__file__).resolve().parents[1] / name)
            for name in ('tools/diagnose_rollback_activity.py', 'app/rollback_activity.py')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshots', 'parent', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
