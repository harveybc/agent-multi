"""Collect source receipts and reproduce a COT producer defect, never model input."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def collect(bundle, recovery, producer, output):
    if sha(bundle / 'model.pt') != '00f2fc65a39f9d5a53572ef915f433b57d5f7939b54699ad34e4274b0ac7874f':
        raise ValueError('ACTOR_IDENTITY')
    if sha(bundle / 'RECIPE.json') != '5cefbb0b3152c1e69926ac2859d9655e6a973563c12420fe5fe4267459654718':
        raise ValueError('RECIPE_IDENTITY')
    if sha(producer) != '8b6e203e2d8b5cd4db415f7366b4bbca2b67f70fb978ea12126f2281a072e0af':
        raise ValueError('PRODUCER_IDENTITY')
    recipe = read(bundle / 'RECIPE.json')
    columns = recipe['tensor']['columns']
    assert len(columns) == 72 and len({c['store_column'] for c in columns}) == 72
    artifacts = {
        'stl_repair': recovery / 'stl_november_v3/RESULT.json',
        'prehistory': recovery / 'warmup_candidate_v3/RESULT.json',
        'activity': recovery / 'activity/consumer_recovered/RESULT.json',
    }
    stl, warm, activity = (read(artifacts[k]) for k in ('stl_repair', 'prehistory', 'activity'))
    spec = importlib.util.spec_from_file_location('pinned_cot_producer', producer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = pd.DataFrame(dict(Date=['2024-01-02'] * 2,
        CFTC_Contract_Market_Code=['099741', '299741'],
        EUR_Long=[200, 20], EUR_Short=[100, 25], EUR_Net_Long=[100, -5]))
    forward = module.cot_features(fixture)['cot.eur.net_long'].tolist()
    reverse = module.cot_features(fixture.iloc[::-1])['cot.eur.net_long'].tolist()
    assert forward == [-5.0] and reverse == [100.0]
    output.mkdir(parents=True, exist_ok=True)
    documents = {
        'STATUS.json': dict(schema='stl_live_readiness_audit.v1',
            recipe=recipe['identity'], model_sha256=sha(bundle / 'model.pt'),
            source_receipts={k:sha(v) for k,v in artifacts.items()},
            stl=dict(recovered_windows=stl['recovered_original_windows'],
                empty_months=stl['remaining_empty_months'],
                training_released=stl['training_released'], current_actor_uses_stl=False),
            prehistory=dict(reconstructed=warm['reconstructed_columns'],
                unresolved=warm['unresolved_columns'], rows=warm['pre_train_rows'],
                admitted=warm['new_store_admitted']),
            activity={k:activity[k] for k in ('status','rows','weights_sha256','maximum_error','reload_exact','live_ready')},
            live_ready=False, live_feed_receipt=None, original_cot_csv_reverified=False,
            broker_configuration_changed=False),
        'CURRENT_RECIPE_INPUTS.json': dict(schema='selected72_live_handoff.v1',
            recipe=recipe['identity'], recipe_sha256=sha(bundle / 'RECIPE.json'),
            model_sha256=sha(bundle / 'model.pt'), window=recipe['tensor']['window'],
            columns=columns, live_readiness='NOT_VERIFIED', promotion_authorized=False,
            cot_columns=[c['store_column'] for c in columns if c['store_column'].startswith('cot.')]),
        'COT_REORDER_WITNESS.json': dict(schema='cot_reorder_witness.v1',
            evidence='SYNTHETIC_UNIT_FIXTURE_ONLY_NOT_TRAINING_OR_LIVE_DATA',
            producer_sha256=sha(producer), fixture=fixture.to_dict(orient='records'),
            net_forward=forward, net_reversed=reverse,
            finding='duplicate-date values depend on input order; market code is not filtered',
            historical_impact_quantified=False),
    }
    for name, value in documents.items():
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    (output / 'MANIFEST.json').write_text(json.dumps(dict(
        files={name:sha(output / name) for name in documents}), indent=2) + '\n')
    print(json.dumps(dict(columns=len(columns), recovered_windows=stl['recovered_original_windows'],
                         unresolved=warm['unresolved_columns'], live_ready=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'recovery', 'producer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    collect(args.bundle, args.recovery, args.producer, args.output)
