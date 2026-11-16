"""Replay a sealed checkpoint on one device, exporting logits for cross-device audit."""
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
from tools.run_discrete_curve_probe import configure, read, write


def run(a):
    source_sha = file_sha(Path(__file__))
    configure(7)
    # Match the long-run producer's convolution precision, not just GEMM settings.
    torch.backends.cudnn.allow_tf32 = False
    if a.disable_attention_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
    if a.output.exists():
        raise ValueError('NEW_OUTPUT_REQUIRED')
    model, _ = load_bundle(a.bundle, a.device)
    weight_sha = file_sha(a.weights)
    if weight_sha != a.expected_sha:
        raise ValueError('CHECKPOINT_SHA_MISMATCH')
    model.loss.load_state_dict(torch.load(a.weights, map_location=a.device, weights_only=True))
    parent = read(a.parent/'SPEC.json')
    design = read(a.design); risk = read(a.parent/'RISK.json')
    market = market_for(a.parent, parent, model.recipe)
    before = tree_digest(model.loss.state_dict())
    logits, _ = emit(model, market.inputs)
    prepared = prepared_signals(logits)
    metrics = {p: evaluate(environment(market, risk, design['horizon'], interval,
                       model.effective['gamma']), logits, prepared=prepared)
               for p, interval in model.recipe['split']['periods'].items()}
    if before != tree_digest(model.loss.state_dict()):
        raise ValueError('PROBE_MUTATED_WEIGHTS')
    if source_sha != file_sha(Path(__file__)):
        raise ValueError('PROBE_SOURCE_CHANGED_DURING_EXECUTION')
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'LOGITS.npz', logits=logits, actions=prepared[0],
                        valid=prepared[3], periods=market.inputs.period.astype(str))
    write(a.output/'RESULT.json', dict(device=a.device, weight_sha256=weight_sha,
        support_sha256=support_hash(market.inputs), metrics=metrics,
        torch_version=torch.__version__, source_sha256=source_sha, trained=False,
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
    print({p: {k: m[k] for k in ('return_pct', 'trades')} for p, m in metrics.items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'weights', 'design', 'parent', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--expected-sha', required=True)
    p.add_argument('--disable-attention-fastpath', action='store_true')
    run(p.parse_args())
