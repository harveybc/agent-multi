"""Explicit genotype/phenotype and append-only lineage for the post-phase2 search."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path

from app.rl_subset import digest


def value_at(config, path):
    value = config
    for key in path.split('.'):
        value = value[_path_key(value, key)]
    return value


def set_at(config, path, value):
    parts = path.split('.')
    node = config
    for key in parts[:-1]:
        node = node[_path_key(node, key)]
    node[_path_key(node, parts[-1])] = value


def _path_key(node, key):
    if isinstance(node, list):
        if not key.isdigit() or str(int(key)) != key or int(key) >= len(node):
            raise ValueError('GENE_LIST_INDEX')
        return int(key)
    return key


def validate_space(space):
    genes = space['genes']
    paths = [g['path'] for g in genes]
    if not genes or len(set(paths)) != len(paths):
        raise ValueError('GENE_PATHS')
    for g in genes:
        if g['kind'] == 'choice':
            values = g['values']
            if not values or len(set(values)) != len(values):
                raise ValueError('GENE_CHOICES')
        elif g['kind'] in ('linear', 'log10'):
            lo, hi = g['bounds']
            if not all(math.isfinite(x) for x in (lo, hi)) or not lo < hi:
                raise ValueError('GENE_BOUNDS')
            if g['kind'] == 'log10' and lo <= 0:
                raise ValueError('GENE_LOG_DOMAIN')
        else:
            raise ValueError('GENE_KIND')


def encode(space, config):
    validate_space(space)
    result = []
    for g in space['genes']:
        value = value_at(config, g['path'])
        if g['kind'] == 'choice':
            result.append((g['values'].index(value) + .5) / len(g['values']))
        else:
            lo, hi = g['bounds']
            if not lo <= value <= hi:
                raise ValueError('GENE_ENCODE_BOUNDS')
            if g['kind'] == 'log10':
                lo, hi, value = map(math.log10, (lo, hi, value))
            result.append((value - lo) / (hi - lo))
    return result


def decode(space, genome, base):
    validate_space(space)
    if len(genome) != len(space['genes']) or any(
            type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 for x in genome):
        raise ValueError('GENE_DOMAIN')
    effective = copy.deepcopy(base)
    for g, x in zip(space['genes'], genome):
        if g['kind'] == 'choice':
            value = g['values'][min(int(x * len(g['values'])), len(g['values']) - 1)]
        else:
            lo, hi = g['bounds']
            if x in (0, 1):
                value = (lo, hi)[int(x)]
            elif g['kind'] == 'log10':
                value = 10 ** (math.log10(lo) + x * (math.log10(hi) - math.log10(lo)))
            else:
                value = lo + x * (hi - lo)
        set_at(effective, g['path'], value)
    attn = effective['architecture']['temporal_attention']
    if attn['embedding'] % attn['heads']:
        raise ValueError('GENE_HEAD_DIVISIBILITY')
    return effective


def read_ledger(path):
    rows, previous = [], None
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        body = {k: v for k, v in row.items() if k != 'sha256'}
        if (row['sha256'] != digest(body) or row['previous_sha256'] != previous or
                row['sequence'] != len(rows)):
            raise ValueError('GENE_LEDGER_CHAIN')
        previous = row['sha256']
        rows.append(row)
    return rows


def append_event(path, event):
    required = {'campaign_id', 'candidate_id', 'stage', 'status', 'raw_genome',
                'decoded_config', 'seed', 'parents', 'metrics', 'artifacts'}
    if not required <= set(event):
        raise ValueError('GENE_LEDGER_FIELDS')
    if event['status'] not in {'REQUESTED', 'COMPLETE', 'REJECTED', 'FAILED', 'CACHE_HIT'}:
        raise ValueError('GENE_LEDGER_STATUS')
    # No NaN/Infinity, including in nested metrics; failure is a typed event.
    json.dumps(event, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = read_ledger(path)
        body = dict(sequence=len(rows), previous_sha256=rows[-1]['sha256'] if rows else None,
                    event=event)
        row = dict(body, sha256=digest(body))
        with path.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return row
