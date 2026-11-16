"""Authenticated SAC parameter rollback, without rewinding replay or scheduler."""
import json
from pathlib import Path

import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.optimizer_layout import optimizer_layout
from app.structural_refit import verify_files
from app.training_forensics import tree_digest


ROLLBACK_PLATEAU = dict(factor=.5, threshold=1e-8, cooldown=0, min_actor_lr=1e-7,
    post_reduction_epochs=10, exact_patience_divisor=2, restore_best=True, stop_before_reduce=True)


def restore_validation_best(model, output, state, reduced_rates):
    best = state.get('best_bundle')
    if not best:
        raise ValueError('ROLLBACK_NO_ELIGIBLE_BEST')
    root = (Path(output)/'bundles').resolve()
    bundle = (Path(output)/best['relative_path']).resolve()
    if bundle.parent != root:
        raise ValueError('ROLLBACK_BUNDLE_PATH')
    if (file_sha(bundle/'MANIFEST.json') != best['manifest_sha256']
            or file_sha(bundle/'model.pt') != best['model_sha256']):
        raise ValueError('ROLLBACK_BUNDLE_HASH')
    manifest = json.loads((bundle/'MANIFEST.json').read_text())
    verify_files(bundle, manifest['files'])
    if manifest['checkpoint']['score'] != state['stopping']['best']:
        raise ValueError('ROLLBACK_BEST_SCORE')
    payload = torch.load(bundle/'model.pt', map_location=model.device, weights_only=False)
    if (payload['recipe_identity'] != model.recipe['identity']
            or payload['effective'] != model.effective
            or payload['codec'] != 'NOP_BUY_SELL_CLOSE_OPPOSITE_V1'
            or payload['optimizer_parameter_names'] != optimizer_layout(model)
            or payload['trainability'] != {n:p.requires_grad for n,p in model.loss.named_parameters()}):
        raise ValueError('ROLLBACK_MODEL_CONTRACT')
    current = model.loss.state_dict()
    if set(current) != set(payload['loss']) or any(current[n].shape != v.shape for n,v in payload['loss'].items()):
        raise ValueError('ROLLBACK_TENSOR_LAYOUT')
    if (len(reduced_rates) != len(model.optimizer.param_groups)
            or not np.isfinite(reduced_rates).all() or min(reduced_rates) <= 0):
        raise ValueError('ROLLBACK_RATES')
    before = tree_digest(current)
    # Keep the optimizer object: the scheduler must not point at a discarded one.
    model.loss.load_state_dict(payload['loss'])
    model.optimizer.load_state_dict(payload['optimizer'])
    for group, rate in zip(model.optimizer.param_groups, reduced_rates):
        group['lr'] = float(rate)
    after = tree_digest(model.loss.state_dict())
    if (after != tree_digest(payload['loss']) or tree_digest(model.optimizer.state_dict()['state'])
            != tree_digest(payload['optimizer']['state'])):
        raise ValueError('ROLLBACK_STATE_PARITY')
    return dict(restored_epoch=manifest['checkpoint']['epoch'], global_epoch=state['epoch'],
        manifest_sha256=best['manifest_sha256'], model_sha256=best['model_sha256'],
        before_weights_sha256=before, after_weights_sha256=after, rates=list(reduced_rates),
        moments_restored=True, targets_and_temperature_restored=True,
        replay_rewound=False, random_streams_rewound=False, global_patience_reset=False)
