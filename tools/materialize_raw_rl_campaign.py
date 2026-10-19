"""Bind an admitted raw-input bundle and pinned runtimes to the static RL queue."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from tools.verify_rl_static_engineering import save_status


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def commit(root):
    return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def materialize(design_path, input_root, admission_root, gym_root, predictor_root,
                output_root, gpu_evidence, gpu_uuid, warehouse_url='http://127.0.0.1:5057'):
    from app.curated_rl_input import validate_curated_rl_input
    from tools.curated_rl_campaign import load_contract
    design = json.loads(design_path.read_text())
    if (design.get('schema') != 'agent_multi.raw_baseline_design.v1'
            or design.get('scope') != 'EURUSD_CORE_PRICES_ONLY_NOT_FINAL_FEATURE_SELECTION'
            or design.get('refit_cadence') != 'NONE'):
        raise ValueError('STATIC_RAW_BASELINE_DESIGN_REQUIRED')
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((input_root/'RECEIPT.json').read_text())
    admission_file = admission_root/'ADMISSION.json'
    admission = json.loads(admission_file.read_text())
    if (receipt.get('status') != 'PREPARED_AND_VALIDATED' or receipt.get('scope') != 'RAW_BASELINE'
            or receipt.get('feature_selection_complete') is not False
            or receipt['owner_admission_sha256'] != digest(admission_file)
            or admission.get('scope') != 'CORE_PRICES_ONLY' or admission.get('admitted') is not True):
        raise ValueError('OWNER_ADMITTED_RAW_INPUT_REQUIRED')
    closures = admission_root/admission['closure_intervals_file']
    if (closures.resolve().parent != admission_root.resolve()
            or digest(closures) != admission['closure_intervals_sha256']):
        raise ValueError('ADMITTED_CLOSURE_CALENDAR_DRIFT')
    common, splits = copy.deepcopy(design['common_config']), {}
    for role in ('train', 'validation'):
        item = receipt['splits'][role]
        manifest = input_root/item['manifest_file']
        data = input_root/item['csv_file']
        if any(p.resolve().parent != input_root.resolve() for p in (manifest, data)):
            raise ValueError('INPUT_MEMBER_OUTSIDE_BUNDLE')
        columns = json.loads(manifest.read_text())['feature_columns']
        split = dict(input_data_file=str(data.resolve()), curated_input_manifest=str(manifest.resolve()),
            curated_input_manifest_sha256=digest(manifest), curated_input_split=role, feature_columns=columns)
        identity = validate_curated_rl_input({**common, **split, 'require_curated_rl_input': True, 'max_rows': None})
        if identity != item['validator_result']:
            raise ValueError('RAW_BUNDLE_VALIDATOR_IDENTITY_DRIFT')
        splits[role] = split
    common['session_calendar_intervals'] = json.loads(closures.read_text())
    common['session_exposure_policy']['calendar_identity'] = admission['binding']['calendar_sha256']
    common['session_account_fingerprint'] = hashlib.sha256(b'raw-eurusd-analytical-simulator-v1').hexdigest()
    body = dict(schema='curated_rl_campaign.v1', classification='NON_GOVERNING',
        scientific_scope='REAL_DATA_DEVELOPMENT', execution_backend='local_subprocess',
        seeds=design['seeds'], arms=design['arms'], common_config=common, splits=splits,
        input_policy='RAW_BASELINE_NOT_FINAL_FEATURE_SELECTION', design_sha256=digest(design_path),
        cost_scope=design['cost_scope'],
        input_receipt_sha256=digest(input_root/'RECEIPT.json'),
        execution=dict(device='cuda', gym_root=str(gym_root.resolve()), gym_code_commit=commit(gym_root),
            task_timeout_seconds=dict(train=14400, validation=3600), max_attempts_per_task=2,
            gpu_evidence=dict(path=str(gpu_evidence.resolve()), sha256=digest(gpu_evidence),
                assigned_gpu_uuid=gpu_uuid, max_age_seconds=86400)),
        publication=dict(mode='WAREHOUSE_REPORTS', predictor_root=str(predictor_root.resolve()),
            predictor_code_commit=commit(predictor_root), code_commit=commit(root),
            actor='canonical-rl-curation', lake='curated-core-prices',
            warehouse_url=warehouse_url, token_env='DATA_GOV_LAKE_TOKEN', pump_between_stages=True,
            experiment_set_key='eurusd-raw-static-20261019',
            resources={role:'raw-rl-eurusd/'+role+'.csv' for role in splits}))
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root/'CONTRACT.json'
    if path.exists():
        if json.loads(path.read_text()) != body:
            raise ValueError('EXISTING_CAMPAIGN_CONTRACT_DIFFERS')
    else:
        save_status(path, body)
    load_contract(path)
    save_status(output_root/'MATERIALIZATION.json', dict(schema='raw_rl_materialization.v1',
        state='READY', contract_sha256=digest(path), feature_selection_complete=False,
        train_rows=receipt['splits']['train']['validator_result']['row_count'],
        validation_rows=receipt['splits']['validation']['validator_result']['row_count']))
    return path


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('input-root', 'admission-root', 'gym-root', 'predictor-root', 'output-root', 'gpu-evidence'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--warehouse-url', required=True)
    p.add_argument('--design', type=Path, default=Path(__file__).resolve().parents[1]/'examples/config/rl_raw_eurusd_static.v1.json')
    a = p.parse_args()
    path = materialize(a.design, a.input_root, a.admission_root, a.gym_root, a.predictor_root,
                       a.output_root, a.gpu_evidence, a.gpu_uuid, a.warehouse_url)
    print(json.dumps(dict(contract=str(path), sha256=digest(path), state='READY')))


if __name__ == '__main__':
    main()
