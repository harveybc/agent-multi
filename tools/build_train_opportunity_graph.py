#!/usr/bin/env python3
"""Recover pinned indicator bytes and expand immutable TRAIN K=3 receipts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.zigzag_opportunities import load_tape, engine, sha
from app.train_opportunity_graph import expand_edges
from tools.zigzag_opportunity_deap import authenticated, write


def pinned_blob(repo, name, digest):
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('UNSAFE_SOURCE_NAME')
    p = repo / name
    if p.is_file() and sha(p) == digest:
        return p.read_bytes(), 'CURRENT_BYTES_MATCH'
    revisions = subprocess.check_output(['git','-C',str(repo),'log','--all','--format=%H','--',name],text=True).split()
    for rev in revisions:
        result = subprocess.run(['git','-C',str(repo),'show',f'{rev}:{name}'],capture_output=True)
        if result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == digest:
            return result.stdout, rev
    raise ValueError('PIN_UNRECOVERABLE:'+name)


def build(source, output):
    if output.exists() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('NEW_OUTPUT_OUTSIDE_SOURCE_REQUIRED')
    spec = json.loads((source/'SPEC.private.json').read_text())
    ident = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    report = authenticated(source/'RESULTS.json',ident)
    if spec['train'] != ['2020-01-01','2024-01-01'] or spec['future_pivots'] != 3 or spec['horizons'] != [3,6,12,24]:
        raise ValueError('TRAIN_GRAPH_CONTRACT')
    if spec['software'] != {p:importlib.metadata.version(p) for p in spec['software']}:
        raise ValueError('SOURCE_SOFTWARE_CHANGED')
    for name,digest in spec['code_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('SOURCE_PRODUCER_CHANGED:'+name)
    payloads = {}
    restored = {}
    for name,digest in spec['engine_sha256'].items():
        payloads[name], revision = pinned_blob(Path(spec['engine_root']),name,digest)
        if revision != 'CURRENT_BYTES_MATCH':
            restored[name] = revision
    output.mkdir(parents=True)
    snapshot = output/'source_engine'
    for name,data in payloads.items():
        path = snapshot/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    runtime_spec = dict(spec,engine_root=str(snapshot))
    tape = load_tape(runtime_spec)
    zigzag,_,_ = engine(snapshot)
    before = {p.name:sha(p) for p in (source/'SPEC.private.json',source/'RESULTS.json')}
    rows = []
    for row in report['rows']:
        path = source/row['records_file']
        if path.resolve().parent != source.resolve() or sha(path) != row['records_sha256']:
            raise ValueError('SOURCE_RECORDS_CHANGED')
        before[path.name] = sha(path)
        with np.load(path,allow_pickle=False) as archive:
            records = {k:archive[k] for k in archive.files}
        params = zigzag.Params(**row['params'])
        pivots = []
        for a,b in tape.chunks:
            pivots.extend((a+i,kind,price) for i,kind,price in zigzag.hindsight_pivots(tape.high[a:b],tape.low[a:b],params))
        graph,summary = expand_edges(tape,pivots,records)
        path = output/f'h{row["horizon"]}.npz'
        np.savez_compressed(path,**graph)
        rows.append(dict(horizon=row['horizon'],params=row['params'],graph_file=path.name,
                         graph_sha256=sha(path),**summary))
        print(row['horizon'],summary,flush=True)
    if any(sha(source/name) != digest for name,digest in before.items()):
        raise ValueError('SOURCE_CHANGED_DURING_GRAPH')
    result = dict(schema='train_k3_opportunity_graph.v1',source_identity=ident,
        source_sha256=before,input_sha256=spec['input_sha256'],engine_sha256=spec['engine_sha256'],
        recovered_source_revisions=restored, train=spec['train'], source_unchanged=True,
        scope='ALL_K3_HINDSIGHT_REFERENCE_EDGES_NOT_EXECUTABLE_FILLS_OR_ACCOUNT_RESULTS',
        code_sha256={p:sha(ROOT/p) for p in ('app/train_opportunity_graph.py','tools/build_train_opportunity_graph.py')},
        data=tape.metadata,rows=rows)
    write(output/'GRAPH.json',result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    build(a.source,a.output)
