"""Frozen full-width inference and authenticated cache; no fitting or projection."""
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from app.rl_subset import digest


def masked_emit(model, inputs, keep):
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (inputs.windows.shape[2],):
        raise ValueError('MASK_WIDTH')
    logits = np.full((len(inputs.valid), 3), np.nan, np.float32)
    indices = np.flatnonzero(inputs.valid)
    with torch.inference_mode():
        for start in range(0, len(indices), 256):
            ix = indices[start:start+256]
            x = torch.tensor(inputs.windows[ix], device=model.device)
            x[:, :, ~keep] = 0
            logits[ix] = model.surfaces(x).cpu().numpy()
    return logits


def prepared_signals(logits):
    """Vectorize market-only policy diagnostics, never the account transition."""
    finite = np.isfinite(logits).all(axis=1)
    # Invalid unused rows remain explicitly invalid, not imputed observations.
    raw = np.where(finite[:, None], logits, 0)
    maximum = raw.max(axis=1, keepdims=True)
    ties = np.isclose(raw, maximum, atol=1e-10, rtol=0)
    actions = np.where(ties.sum(axis=1) == 1, ties.argmax(axis=1), 0)
    p = np.exp(raw-maximum); p = p/p.sum(axis=1, keepdims=True)
    entropy = -np.sum(p*np.log(np.maximum(p, 1e-30)), axis=1)
    commands = np.empty((len(raw), 3), dtype=entropy.dtype)
    commands[:, 1] = entropy
    for direction in (-1, 1):
        pc = np.stack((p[:, 0]+p[:, 1 if direction == 1 else 2],
                       p[:, 2 if direction == 1 else 1]), axis=1)
        commands[:, direction+1] = -np.sum(pc*np.log(np.maximum(pc, 1e-30)), axis=1)
    return actions, entropy, commands, finite


def cached_measurement(root, context, columns, compute):
    from tools.run_discrete_curve_probe import write
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(context=context, columns=columns))
    path = root/(identity+'.json')
    with (root/(identity+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            row = json.loads(path.read_text())
            if (row.get('identity') != identity or row.get('columns') != columns
                    or row.get('sha256') != digest(row.get('result'))):
                raise ValueError('MASK_CACHE_CHANGED')
            return row['result']
        result = compute()
        write(path, dict(identity=identity, columns=columns, result=result, sha256=digest(result)))
        return result
