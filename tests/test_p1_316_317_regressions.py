"""Regression fixtures ordered by MUSASHI_TO_GENERAL_SATOSHI_POST_P1
_BASELINE_EXECUTION_ORDER §2.3: path relocation, changed contract
content, absent state map, changed tensor, duplicate/missing arm."""
import hashlib
import json
from pathlib import Path

import pytest

import tools.aggregate_l1_curriculum_campaign as agg
from tools.l1_curriculum_experiment import arm_contracts

SEEDS = agg.SEEDS
ARMS = agg.ARMS
STATE = {f"tensor{i}": f"{i:064x}" for i in range(4)}


def _write_contract(tmp, content='{"roles": {"fit_train": {}}}',
                    name="eth_nested_split_contract_v1.json"):
    p = tmp / name
    p.write_text(content)
    return p


def _record(contract_sha, arm, state=STATE, manifest_sha=None):
    return {
        "schema": "agent_multi.l1_curriculum_arm.v3",
        "outcome": "ARM_COMPLETE", "normal_accepted": True,
        "contracts": {"pair_contract": {
            "seed": "S", "nested_split_contract_sha256": contract_sha}},
        "outer_endpoint": {"role": "outer_validation_2024",
                           "scored_rows": 2196,
                           "primary_score_risk_adjusted_return": -0.5},
        "selected_manifest": {"sha256": manifest_sha or "f" * 64,
                              "named_state_sha256": dict(state)},
    }


def _campaign(tmp, mutate=None):
    contract = _write_contract(tmp)
    csha = hashlib.sha256(contract.read_bytes()).hexdigest()
    reports = tmp / "reports"; manifests = tmp / "manifests"
    reports.mkdir(exist_ok=True); manifests.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            man = {"named_state_sha256": dict(STATE)}
            body = json.dumps(man, sort_keys=True)
            msha = hashlib.sha256(body.encode()).hexdigest()
            (manifests / f"seed{seed}_{arm}_selected_manifest.json"
             ).write_text(body)
            rec = _record(csha, arm, manifest_sha=msha)
            (reports / f"seed{seed}_{arm}_report.json").write_text(
                json.dumps(rec))
    if mutate:
        mutate(reports, manifests)
    return reports, manifests, contract


def test_valid_v3_campaign_aggregates(tmp_path):
    r, m, c = _campaign(tmp_path)
    out = agg.aggregate(r, m, c)
    assert out["reports_complete"] == 12
    for arm in ("EN-W", "EN-F"):
        assert out["results"][arm]["informative_easy_seeds"] == 0


# --- 1. path relocation --------------------------------------------------

def test_path_relocation_yields_identical_pair_identity(tmp_path):
    # P1-316: same contract CONTENT at two different absolute paths must
    # produce the same pair identity hash.
    c1 = _write_contract(tmp_path / "a", name="c.json") if (
        (tmp_path / "a").mkdir() or True) else None
    (tmp_path / "b").mkdir()
    c2 = tmp_path / "b" / "c.json"
    c2.write_text(c1.read_text())
    eff1 = {"x": 1, "nested_split_contract": str(c1)}
    eff2 = {"x": 1, "nested_split_contract": str(c2)}
    a1 = arm_contracts(eff1, "N")
    a2 = arm_contracts(eff2, "N")
    assert a1["pair_contract_sha256"] == a2["pair_contract_sha256"]
    assert "nested_split_contract" not in a1["pair_contract"]


# --- 2. changed contract content ----------------------------------------

def test_changed_contract_content_refused(tmp_path):
    r, m, c = _campaign(tmp_path)
    c.write_text('{"roles": {"fit_train": {"MUTATED": true}}}')
    with pytest.raises(agg.AggregationError, match="different nested"):
        agg.aggregate(r, m, c)


def test_changed_contract_changes_pair_identity(tmp_path):
    c1 = _write_contract(tmp_path, name="c1.json")
    c2 = _write_contract(tmp_path, content='{"roles": "OTHER"}',
                         name="c2.json")
    a1 = arm_contracts({"nested_split_contract": str(c1)}, "N")
    a2 = arm_contracts({"nested_split_contract": str(c2)}, "N")
    assert a1["pair_contract_sha256"] != a2["pair_contract_sha256"]


# --- 3. absent state map -------------------------------------------------

def test_absent_embedded_state_map_refused(tmp_path):
    def mutate(reports, manifests):
        f = reports / "seed101_N_report.json"
        rec = json.loads(f.read_text())
        del rec["selected_manifest"]
        f.write_text(json.dumps(rec))
    r, m, c = _campaign(tmp_path, mutate)
    with pytest.raises(agg.AggregationError, match="without embedded"):
        agg.aggregate(r, m, c)


def test_malformed_empty_state_map_refused(tmp_path):
    def mutate(reports, manifests):
        f = reports / "seed202_EN-W_report.json"
        rec = json.loads(f.read_text())
        rec["selected_manifest"]["named_state_sha256"] = {}
        f.write_text(json.dumps(rec))
    r, m, c = _campaign(tmp_path, mutate)
    with pytest.raises(agg.AggregationError, match="malformed embedded"):
        agg.aggregate(r, m, c)


def test_evidence_hash_mismatch_refused(tmp_path):
    def mutate(reports, manifests):
        # evidence file mutated AFTER the report bound its digest
        f = manifests / "seed303_EN-F_selected_manifest.json"
        f.write_text(json.dumps({"named_state_sha256": dict(STATE)},
                                sort_keys=True, indent=1))
    r, m, c = _campaign(tmp_path, mutate)
    with pytest.raises(agg.AggregationError, match="hash mismatch"):
        agg.aggregate(r, m, c)


# --- 4. changed tensor ---------------------------------------------------

def test_changed_tensor_counts_as_divergence(tmp_path):
    def mutate(reports, manifests):
        for name in ("seed404_EN-F_report.json",):
            f = reports / name
            rec = json.loads(f.read_text())
            st = dict(STATE); st["tensor0"] = "e" * 64
            man = {"named_state_sha256": st}
            body = json.dumps(man, sort_keys=True)
            rec["selected_manifest"] = {
                "sha256": hashlib.sha256(body.encode()).hexdigest(),
                "named_state_sha256": st}
            f.write_text(json.dumps(rec))
            (manifests / "seed404_EN-F_selected_manifest.json"
             ).write_text(body)
    r, m, c = _campaign(tmp_path, mutate)
    out = agg.aggregate(r, m, c)
    row = next(x for x in out["rows"]
               if x["seed"] == 404 and x["arm"] == "EN-F")
    assert row["easy_treatment_diverged"] is True
    assert out["results"]["EN-F"]["informative_easy_seeds"] == 1


# --- 5. duplicate / missing arm -----------------------------------------

def test_missing_arm_refused(tmp_path):
    def mutate(reports, manifests):
        (reports / "seed101_EN-W_report.json").unlink()
    r, m, c = _campaign(tmp_path, mutate)
    with pytest.raises(agg.AggregationError, match="missing report"):
        agg.aggregate(r, m, c)


def test_duplicate_arm_content_refused(tmp_path):
    def mutate(reports, manifests):
        # an EN-W report smuggled in as EN-F (duplicate content for a
        # different arm) — caught because the aggregator addresses arms
        # by filename and pair identity, and EN-F's report now claims a
        # different pair seed marker
        src = json.loads((reports / "seed101_EN-W_report.json").read_text())
        src["contracts"]["pair_contract"]["seed"] = "OTHER"
        (reports / "seed101_EN-F_report.json").write_text(json.dumps(src))
    r, m, c = _campaign(tmp_path, mutate)
    with pytest.raises(agg.AggregationError, match="pair identity"):
        agg.aggregate(r, m, c)
