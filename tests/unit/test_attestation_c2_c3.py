"""C2/C3 adversarial batteries (order @1649e7c0 §§3-4): the external
auditor refuses every forgery class and the SAC gate is DERIVED from
the bound report/audit, never declared."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


auditor = _load("ext_audit", REPO / "tools/screen_v2_external_audit.py")
gate_mod = _load("sac_gate", REPO / "tools/sac_scientific_gate.py")

from agent_plugins.experiment_runtime import (  # noqa: E402
    RunDirectory, unit_id)


@pytest.fixture()
def tiny_run(tmp_path):
    """A minimal COMPLETE single-phase run the auditor can accept,
    built with the CURRENT (corrected) runtime so results carry
    unit_id."""
    home = Path.home() / ".cache" / "att_tests" / uuid.uuid4().hex
    home.mkdir(parents=True)
    run = RunDirectory(home / "round1")
    ident = {"experiment": "e", "family": "f", "window": 32,
             "latent": 16, "budget": 300, "seed": 1, "origin": 1,
             "treatment": "cell"}
    uid = unit_id(ident)
    run.write_ledger({"schema": "s", "experiment": "e",
                      "units": [{"unit_id": uid, "identity": ident}],
                      "digests": {}, "campaign_wall_ceiling_s": 60,
                      "unit_timeout_s": 30})
    run.claim(uid, expected_digests={})
    run.release(uid, "COMPLETED",
                result={"family": "f", "window": 32, "latent": 16,
                        "budget": 300, "monitor_r2": -0.1,
                        "calibration_r2": 0.0}, attempt=1)
    yield home, run, uid, ident
    shutil.rmtree(home, ignore_errors=True)


def _mini_audit(home):
    """Run ONLY the per-phase unit checks of the auditor against the
    tiny run by invoking audit() with the repo as code root (report
    absent -> incomplete verdict; the FINDINGS list is what these
    tests assert on)."""
    return auditor.audit(
        home, REPO,
        REPO / "docs/audits/evidence/"
               "POSITIVE_SKILL_SCREEN_PREDECLARATION_V2_"
               "2026_09_03.json")


class TestAuditorRefusals:

    def test_clean_tiny_run_has_no_unit_findings(self, tiny_run):
        home, _run, _uid, _ident = tiny_run
        result = _mini_audit(home)
        unit_findings = [f for f in result["findings"]
                        if f["kind"] not in ("code_digest_mismatch",
                                             "config_digest_mismatch")]
        assert unit_findings == []

    def test_forged_state_identity_refuses(self, tiny_run):
        home, run, uid, _ident = tiny_run
        state_path = run.root / "units" / f"{uid}.state.json"
        state = json.loads(state_path.read_text())
        state["identity"]["treatment"] = "forged_treatment"
        state_path.write_text(json.dumps(state))
        kinds = {f["kind"] for f in _mini_audit(home)["findings"]}
        assert "state_identity_forged" in kinds

    def test_non_completed_unit_is_a_finding(self, tiny_run):
        home, run, uid, _ident = tiny_run
        state_path = run.root / "units" / f"{uid}.state.json"
        state = json.loads(state_path.read_text())
        state["state"] = "FAILED"
        state_path.write_text(json.dumps(state))
        (run.root / "units" / f"{uid}.result.json").unlink()
        kinds = {f["kind"] for f in _mini_audit(home)["findings"]}
        assert "non_completed_unit" in kinds

    def test_foreign_result_and_log_files_are_findings(self,
                                                       tiny_run):
        home, run, _uid, _ident = tiny_run
        (run.root / "units" / f"{'e' * 20}.result.json").write_text(
            "{}")
        (run.root / "units" / f"{'e' * 20}.attempt1.log").write_text(
            "x")
        kinds = {f["kind"] for f in _mini_audit(home)["findings"]}
        assert "foreign_result_file" in kinds
        assert "foreign_log_file" in kinds

    def test_unit_id_recompute_enforced(self, tiny_run):
        home, run, uid, ident = tiny_run
        # forge a ledger whose identity does not hash to its unit id
        ledger = run.ledger()
        ledger["units"][0]["identity"]["seed"] = 999
        ledger["ledger_digest"] = auditor.sha_obj(
            {k: v for k, v in ledger.items()
             if k != "ledger_digest"})
        (run.root / "ledger.json").write_text(json.dumps(ledger))
        kinds = {f["kind"] for f in _mini_audit(home)["findings"]}
        assert ("unit_id_does_not_recompute" in kinds
                or "state_identity_forged" in kinds)

    def test_unknown_digest_key_is_a_finding(self, tiny_run):
        home, run, _uid, _ident = tiny_run
        ledger = run.ledger()
        ledger["digests"]["mystery_key"] = "a" * 64
        ledger["ledger_digest"] = auditor.sha_obj(
            {k: v for k, v in ledger.items()
             if k != "ledger_digest"})
        (run.root / "ledger.json").write_text(json.dumps(ledger))
        kinds = {f["kind"] for f in _mini_audit(home)["findings"]}
        assert "unknown_digest_key" in kinds

    def test_frozen_commit_mismatch_is_a_finding(self, tiny_run):
        home, _run, _uid, _ident = tiny_run
        result = auditor.audit(
            home, REPO,
            REPO / "docs/audits/evidence/"
                   "POSITIVE_SKILL_SCREEN_PREDECLARATION_V2_"
                   "2026_09_03.json",
            frozen_commit="0" * 40)
        kinds = {f["kind"] for f in result["findings"]}
        assert "code_root_not_at_frozen_commit" in kinds


class TestGateDerivation:

    @pytest.fixture()
    def bound(self, tmp_path):
        report = {"schema": "agent_multi.positive_skill_screen.v2",
                  "cells": {}, "survivor_decisions": {},
                  "fusion": {"decisions": {
                      "variant_a": "DOES_NOT_ADVANCE"}}}
        rp = tmp_path / "report.json"
        rp.write_text(json.dumps(report, indent=1))
        audit_art = {
            "schema": "agent_multi.screen_v2_external_audit.v1",
            "verdict": ("SCREEN_V2_NEGATIVE_RESULT_ACCEPTED_WITH_"
                        "LEGACY_BINDING_DISCLOSURE"),
            "audited_report_sha256": hashlib.sha256(
                rp.read_bytes()).hexdigest(),
            "findings": []}
        ap = tmp_path / "audit.json"
        ap.write_text(json.dumps(audit_art, indent=1))
        return rp, ap

    def test_derivation_produces_fail_and_verify_refuses(
            self, bound, tmp_path):
        rp, ap = bound
        gate = gate_mod.evaluate(rp, ap)
        assert gate["gate"] == "SAC_GATE_FAIL_NEGATIVE_RESULT"
        gp = tmp_path / "gate.json"
        gp.write_text(json.dumps(gate, indent=1))
        with pytest.raises(SystemExit, match="not launched"):
            gate_mod.verify_gate_for_dispatch(gp)

    def test_edited_fail_to_pass_refuses(self, bound, tmp_path):
        rp, ap = bound
        gate = gate_mod.evaluate(rp, ap)
        gate["gate"] = "SAC_GATE_PASS"
        gp = tmp_path / "forged.json"
        gp.write_text(json.dumps(gate, indent=1))
        with pytest.raises(SystemExit,
                           match="does not equal the recomputed"):
            gate_mod.verify_gate_for_dispatch(gp)

    def test_unknown_or_missing_fields_refuse(self, bound,
                                              tmp_path):
        rp, ap = bound
        gate = gate_mod.evaluate(rp, ap)
        gate["extra_authority"] = True
        gp = tmp_path / "extra.json"
        gp.write_text(json.dumps(gate, indent=1))
        with pytest.raises(SystemExit, match="unknown"):
            gate_mod.verify_gate_for_dispatch(gp)
        gate = gate_mod.evaluate(rp, ap)
        del gate["external_audit_verdict"]
        gp2 = tmp_path / "missing.json"
        gp2.write_text(json.dumps(gate, indent=1))
        with pytest.raises(SystemExit, match="missing"):
            gate_mod.verify_gate_for_dispatch(gp2)

    def test_doctored_report_cannot_mint_a_gate(self, bound,
                                                tmp_path):
        rp, ap = bound
        doctored = json.loads(rp.read_text())
        doctored["fusion"]["decisions"]["variant_a"] = "ADVANCES"
        rp2 = tmp_path / "doctored.json"
        rp2.write_text(json.dumps(doctored, indent=1))
        with pytest.raises(SystemExit,
                           match="not the one the external audit"):
            gate_mod.evaluate(rp2, ap)

    def test_unaccepted_audit_refuses_derivation(self, bound,
                                                 tmp_path):
        rp, _ap = bound
        bad = {"schema": "agent_multi.screen_v2_external_audit.v1",
               "verdict": "SCREEN_V2_RERUN_REQUIRED",
               "audited_report_sha256": hashlib.sha256(
                   rp.read_bytes()).hexdigest()}
        ap2 = tmp_path / "bad_audit.json"
        ap2.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="ACCEPTED runtime"):
            gate_mod.evaluate(rp, ap2)
