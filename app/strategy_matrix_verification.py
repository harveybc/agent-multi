"""Read-only reconstruction of closed synthetic coverage, independent of the generator."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(root):
    root=Path(root).resolve()
    def read(name):return json.loads((root/name).read_text())
    identity=read('IDENTITY.json');close=read('CLOSE.json');summary=read('SUMMARY.json')
    if close['summary']!=summary or summary['state']!='PASS_DECLARED_GRID':
        raise ValueError('MATRIX_NOT_COMPLETE_PASS')
    for name,sha in close['artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=sha:
            raise ValueError('ARTIFACT_HASH:'+name)
    design=identity['design']
    if identity['limit'] is not None or not design:raise ValueError('FULL_MULTISCALE_REQUIRED')
    axes=design['axes'];keys=list(axes);expected={}
    for values in itertools.product(*(axes[k] for k in keys)):
        c={**design['constants'],**dict(zip(keys,values))};expected[digest(c)]=c
    actual={p.stem:p for p in (root/'cells').glob('*.json')}
    if set(actual)!=set(expected):raise ValueError('EXACT_CELL_COVERAGE')
    n=len(expected)
    if n!=design['expected_cells'] or n!=summary['cells'] or n!=identity['expected_cells']:
        raise ValueError('CELL_COUNT')
    counts=Counter();wins=Counter();prescribed=0;absent=0
    axis_counts={k:Counter() for k in keys}
    for key,cell in expected.items():
        row=json.loads(actual[key].read_text())
        if row['cell']!=cell or row['state']!='PASS' or row['identity_sha256']!=digest(identity):
            raise ValueError('CELL_BINDING:'+key)
        for axis in keys:axis_counts[axis][str(cell[axis])]+=1
        for c in row['comparisons']:
            if c['horizon']!=cell['horizon']:raise ValueError('HORIZON_BINDING')
            counts[c['state']]+=1;wins[str(c['chosen'])]+=1
        required=3*sum(c['state']=='FLAT' for c in row['comparisons'])+2*sum(
            c['state'] in ('LONG','SHORT') for c in row['comparisons'])
        if len(row['prescribed'])!=required:raise ValueError('PRESCRIBED_COVERAGE')
        prescribed+=len(row['prescribed']);absent+=len(row['unreachable'])
    if any(counts[k]!=v for k,v in summary['comparisons'].items()) or dict(wins)!=summary['chosen']:
        raise ValueError('SUMMARY_COUNTS')
    if prescribed!=summary['prescribed_checks'] or absent!=summary['unreachable'] or summary['failures']!=0:
        raise ValueError('SUMMARY_CHECKS')
    for axis in keys:
        if any(v!=n//len(axes[axis]) for v in axis_counts[axis].values()):
            raise ValueError('AXIS_BALANCE')
    return dict(schema='independent_multiscale_coverage.v1',state='VERIFIED',cells=n,
        artifacts=len(close['artifacts']),axis_counts={k:dict(v) for k,v in axis_counts.items()},
        comparisons=dict(counts),chosen=dict(wins),prescribed_checks=prescribed,unreachable=absent,
        close_sha256=hashlib.sha256((root/'CLOSE.json').read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),financial_admission=False)
