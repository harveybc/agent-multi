#!/usr/bin/env python3
"""Native BID-anchored finite TRAIN prefixes, all K=3 alternatives retained."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.zigzag_opportunities import engine,sha
from app.train_native_pareto import TrainPrefixWorld,calendar_sample
from tools.calculate_sequential_pareto_reference import independent_check
from tools.zigzag_opportunity_deap import write


def run(source, graph_root, engine_root, output):
    if output.exists():
        raise ValueError('NEW_OUTPUT_REQUIRED')
    spec=json.loads((source/'SPEC.private.json').read_text())
    graph=json.loads((graph_root/'GRAPH.json').read_text())
    if graph['train'] != ['2020-01-01','2024-01-01'] or spec['train'] != graph['train']:
        raise ValueError('TRAIN_ONLY')
    if graph['source_sha256']['SPEC.private.json'] != sha(source/'SPEC.private.json'):
        raise ValueError('SOURCE_GRAPH_IDENTITY')
    for k,digest in graph['input_sha256'].items():
        if sha(spec['inputs'][k]) != digest:
            raise ValueError('INPUT_CHANGED:'+k)
    graphs={}
    for r in graph['rows']:
        path=graph_root/r['graph_file']
        if path.resolve().parent != graph_root.resolve() or sha(path) != r['graph_sha256']:
            raise ValueError('GRAPH_ARRAY_CHANGED')
        with np.load(path,allow_pickle=False) as archive:
            graphs[r['horizon']]={k:archive[k] for k in archive.files}
    _,_,X=engine(engine_root)
    from app.short_horizon.expert_guided.data import calendar_from_closures
    cal=calendar_from_closures(json.loads(Path(spec['inputs']['closures']).read_text()))
    recipe=json.loads(Path(spec['inputs']['recipe']).read_text())
    profile=X.RealisticProfile(**recipe['business']['evaluation_and_policy']['profile'])
    f=pd.read_parquet(spec['inputs']['bars'],filters=[('available_at','>=',pd.Timestamp('2020-01-01',tz='UTC')),
                                                    ('available_at','<',pd.Timestamp('2024-01-01',tz='UTC'))])
    f=f.set_index('timestamp')
    if not np.array_equal(pd.DatetimeIndex(f.available_at).as_unit('ns').asi8,f.index.as_unit('ns').asi8+3600*10**9):
        raise ValueError('AVAILABILITY_CLOCK')
    bars=f[['BidOpen','BidHigh','BidLow','BidClose']].rename(columns=dict(BidOpen='OPEN',BidHigh='HIGH',BidLow='LOW',BidClose='CLOSE'))
    selected=calendar_sample(bars,cal)
    risk_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    risk=json.loads(risk_path.read_text())
    output.mkdir(parents=True)
    cases=[]; replay_count=0; max_error=0.
    for year,(a,b,seg) in sorted(selected.items()):
        world=TrainPrefixWorld(bars.iloc[a:b],seg,a,graphs,risk,profile)
        result,leaves=world.exact()
        independent_check(leaves,result['frontier'])
        for row in result['frontier']:
            actual=world.replay(row['trace'])
            if actual['vector'] != row['vector']:
                raise ValueError('TRAIN_COLD_REPLAY')
            max_error=max(max_error,abs(actual['cash_error']));replay_count+=1
            row['ledger']=actual['trades'];row['plans']=actual['plans']
        best=max(result['frontier'],key=lambda r:r['vector'])
        result.update(year=year,session_open=seg.market_open.isoformat(),max_return=best,
                      complete=True,scope='EXACT_THREE_DECISION_PREFIX_THEN_PLANNED_EXIT_NOT_WEEK_OR_YEAR_OPTIMUM')
        write(output/f'{year}_all_payoffs.json',leaves)
        result['all_payoffs_sha256']=sha(output/f'{year}_all_payoffs.json')
        cases.append(result)
        print(year,result['terminal_accounts'],best['vector'],flush=True)
    out=dict(schema='train_native_prefix_pareto.v1',graph_sha256=sha(graph_root/'GRAPH.json'),
        risk_sha256=sha(risk_path),source_identity=graph['source_identity'],
        native_sha256={str(p.relative_to(engine_root)):sha(p) for p in sorted((engine_root/'app/short_horizon').rglob('*.py'))},
        code_sha256={p:sha(ROOT/p) for p in ('app/train_native_pareto.py','tools/verify_train_prefix_pareto.py',
                     'app/sequential_account_pareto.py','app/dynamic_account_panel.py','app/horizon_risk_rules.py')},
        quote_semantics='OBSERVED_BID_UNCHANGED_ASK_EQUALS_BID_PLUS_ASSUMED_SPREAD',
        selection_rule='FIRST_COMPLETE_CALENDAR_SESSION_PER_TRAIN_YEAR_NO_SCORE_SELECTION',
        cold_frontier_replays=replay_count,max_cash_error_usd=max_error,cases=cases,
        learned_policy=False,full_train_frontier=False,validation_read=False,test_read=False)
    for k,digest in graph['input_sha256'].items():
        if sha(spec['inputs'][k]) != digest:
            raise ValueError('INPUT_CHANGED_DURING_REPLAY:'+k)
    out['inputs_unchanged']=True
    write(output/'REPORT.json',out)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('source','graph-root','engine-root','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args()
    run(a.source,a.graph_root,a.engine_root,a.output)
