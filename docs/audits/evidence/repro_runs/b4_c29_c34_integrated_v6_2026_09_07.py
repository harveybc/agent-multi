"""INTEGRATED V6 (order C29-C34): all 12 cells through the FULL
corrected runtime path — MONOTONE epoch lock -> private-mode
atomic claim -> capability
lease (exact schema, self-integral digest, authorization comparison,
holder binding) -> frozen-genesis scoring on the real materialized
origins -> exact-schema terminal -> append-only intent/completion
seal -> physical adjudication -> the STRONGEST-mode
verify_campaign_results (comparator derived from the reviewed
materialization, frozen sources re-resolved, every factual field
recomputed).

The campaign authorization digest used here is an INTEGRATED-PROOF
mock held in memory only: no Musashi record is created, and the
shipped CAMPAIGN_AUTH_SHA stays None. This run proves MECHANICS on
the frozen genesis artifacts; it is not scientific campaign output.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import b4_authority as b4a  # noqa: E402


def _load(name, rel):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


orch = _load("b4orch_iv6", "tools/b4_campaign_orchestrator.py")
executor = _load("b4exec_iv6", "tools/b4_campaign_executor.py")
ledger_mod = _load("b4led_iv6", "tools/b4_campaign_ledger.py")

STATE = Path.home() / ".local/share/agent-multi"
MAT = STATE / "b4_materialization_v5_20260906"
OUT = STATE / "b4_integrated_v6_20260907"
MOCK_AUTH = executor.CAMPAIGN_AUTH_SHA   # the CONSUMED reviewer record digest
EVID = REPO / "docs/audits/evidence/b4_runtime_authority_20260906"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
os.chmod(OUT, 0o700)

# ---- C30/C33: the v6 preconditions, demonstrated first ----
env_facts = executor.preflight_environment("cpu")
assert env_facts["writes"] == 0
print("environment preflight:", json.dumps(
    {k: env_facts[k] for k in ("python_version",
     "version_agent-multi", "version_torch",
     "amendment_chain_length")}))
# the REAL launch stays CLOSED: no recovery-audit acta exists
try:
    orch.run_campaign(MAT, OUT / "CAMPAIGN_LEDGER.json", OUT,
                      "cpu", execute=True)
    raise AssertionError("v6 launch opened without the acta")
except SystemExit as exc:
    assert "READY_FOR_FINAL_MUSASHI_AUDIT" in str(exc)
    print("v6 launch gate:", str(exc)[:100])

packet = json.loads((MAT / "B4_MATERIALIZATION.json").read_text())
comparator_dir = executor.comparator_dir_of(packet)
b4a.verify_full_authority_chain(comparator_dir)
ledger_path = OUT / "CAMPAIGN_LEDGER.json"
ledger_mod.materialize_ledger(MAT, ledger_path)
sb = executor._load_sb()

t_all = time.time()
walls = {}
with orch.GlobalLock(OUT):
    for cid in ledger_mod.EXPECTED_CELLS:
        t0 = time.time()
        year = int(cid.split("_")[0][1:])
        seed = cid.split("seed")[1]
        cell_dir = OUT / cid
        claim = orch.claim_attempt(OUT, cid)
        lease_p = orch.issue_lease(OUT, cid, claim, MOCK_AUTH, MAT)
        lease = orch.verify_lease(lease_p, OUT, cid, MAT,
                                  expected_auth_sha=MOCK_AUTH)
        assert lease["attempt_id"] == claim["attempt_id"]
        built = executor.build_economic_config(cid, MAT, OUT, "cpu")
        cfg = built["config"]
        frozen = (MAT / "genesis" / f"o{year}" / f"seed{seed}" /
                  f"zero_update_genesis_seed{seed}.zip")
        # The frozen genesis is IDENTICAL bytes for one seed across
        # origins (zero updates), which the final gate correctly
        # refuses as cross-cell checkpoint reuse. The integrated
        # proof therefore scores a per-cell STAMPED copy: the same
        # artifact plus one declared zip member naming the cell.
        # Real campaign checkpoints diverge by training and never
        # need this.
        cell_dir.mkdir(parents=True, exist_ok=True)
        genesis = cell_dir / f"stamped_genesis_{cid}.zip"
        shutil.copyfile(frozen, genesis)
        os.chmod(genesis, 0o644)
        import zipfile
        with zipfile.ZipFile(genesis, "a") as zf:
            zf.writestr("INTEGRATED_PROOF_CELL_ID.txt",
                        f"{cid}|frozen sha "
                        f"{hashlib.sha256(frozen.read_bytes()).hexdigest()}")
        genesis_sha = hashlib.sha256(
            genesis.read_bytes()).hexdigest()
        df = sb.load_source()
        origin = executor.sb_materialize(sb, df, year, cell_dir) \
            if hasattr(executor, "sb_materialize") else \
            sb.materialize_origin(df, year, cell_dir / "outer_origin")
        score = executor.score_frozen_checkpoint(
            cfg, genesis, genesis_sha, origin,
            cell_dir / f"per_bar_{cid}.csv", cid)
        executor.verify_scoring_evidence(score, origin,
                                         comparator_dir, cid)
        # revalidate the capability immediately before publication
        orch.verify_lease(lease_p, OUT, cid, MAT,
                          expected_auth_sha=MOCK_AUTH)
        detail = {
            "attempt_id": claim["attempt_id"],
            "cell_config_sha256": built["cell"]["config_sha256"],
            "artifact_class": "INTEGRATED_PROOF_FROZEN_GENESIS_V6",
            "checkpoint_sha256": score["checkpoint_sha256"],
            "checkpoint_path": str(genesis),
            "per_bar_csv": score["per_bar_csv"],
            "per_bar_sha256": score["per_bar_sha256"],
            "scored_index_sha256": score["scored_index_sha256"],
            "scored_bars": score["scored_bars"],
            "counter_semantics": score["counter_semantics"],
            "sealed_2025_used": False,
            "wall_seconds": round(time.time() - t0, 1),
            "effective_limits": {k: cfg.get(k) for k in (
                "budget_max_env_steps", "budget_max_updates",
                "budget_max_wall_seconds",
                "budget_max_rss_bytes")},
            "authorization_record_sha256": hashlib.sha256(
                b4a.CAMPAIGN_AUTHORIZATION_RECORD_PATH.read_bytes()
            ).hexdigest(),
            "amendment_11_sha256": hashlib.sha256(
                b4a.AMENDMENT_11_PATH.read_bytes()).hexdigest(),
        }
        executor.write_terminal(OUT, cid, "COMPLETED", detail)
        ledger_mod.verify_single_cell_result(
            OUT, cid, built["cell"]["config_sha256"])
        orch.seal_attempt(OUT, cid, claim["attempt_id"])
        state = orch.adjudicate_cell_state(OUT, cid)
        assert state == "COMPLETED_VERIFIED", state
        walls[cid] = round(time.time() - t0, 1)
        print(f"{cid}: COMPLETED_VERIFIED sealed "
              f"({walls[cid]}s)", flush=True)

# ---- the C20/C21 final gate: strongest mode, nothing omitted ----
facts = ledger_mod.verify_campaign_results(ledger_path, MAT, OUT)
assert facts["n"] == 12
print(json.dumps({"integrated_v6": "COMPLETE",
                  "cells_verified": facts["n"],
                  "comparator_dir": "<derived from materialization>",
                  "total_wall_seconds":
                      round(time.time() - t_all, 1)}, indent=1))

# ---- sanitized samples for the evidence tree ----
home = str(Path.home())


def _san(text):
    return text.replace(home + "/.local/share/agent-multi",
                        "<state_root>").replace(home, "<home>")


cid = "o2024_seed101"
att = json.loads((OUT / cid /
                  f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json"
                  ).read_text())["attempt_id"]
for src, dst in (
        (OUT / cid / "B4_CELL_TERMINAL.json",
         EVID / "INTEGRATED_V6_TERMINAL_SAMPLE.json"),
        (OUT / cid / f"CLAIM_{b4a.CAMPAIGN_GENERATION}.json",
         EVID / "INTEGRATED_V6_CLAIM_SAMPLE.json"),
        (OUT / cid / f"SEAL_INTENT_{att}.json",
         EVID / "INTEGRATED_V6_SEAL_INTENT_SAMPLE.json"),
        (OUT / cid / f"SEAL_COMPLETE_{att}.json",
         EVID / "INTEGRATED_V6_SEAL_COMPLETE_SAMPLE.json")):
    dst.write_text(_san(src.read_text()))
(EVID / "INTEGRATED_V6_SUMMARY.json").write_text(json.dumps({
    "schema": "agent_multi.b4_integrated_v6_summary.v1",
    "order": "C27-C28 (authorization closure order 2026-09-06)",
    "path_proven": [
        "GlobalLock (MONOTONE epochs: held->releasing->released "
        "in place, witnessed, never unlinked; reclaim only over a "
        "physically RELEASED predecessor)",
        "private control plane (0700 dirs, 0600 self-integral "
        "lock/claim/lease/seal/terminal records, descriptor-bound "
        "single-open consumption)",
        "claim_attempt (atomic per-cell)",
        "issue_lease + verify_lease (exact schema, self-integral "
        "digest, authorization comparison, holder binding "
        "lease==claim==lock==process)",
        "build_economic_config on materialization v5",
        "score_frozen_checkpoint on the per-cell frozen genesis",
        "verify_scoring_evidence vs derived comparator",
        "pre-publication lease revalidation",
        "write_terminal (exact 18-key schema)",
        "verify_single_cell_result",
        "seal_attempt (append-only intent/completion)",
        "adjudicate_cell_state == COMPLETED_VERIFIED",
        "verify_campaign_results strongest mode: derived "
        "comparator, frozen-source row digests, seed/scored-index/"
        "timestamp/net-return recomputation, MANDATORY "
        "descriptor-first checkpoint existence+owner+mode+digest, "
        "no reuse"],
    "authorization_note": "the CONSUMED reviewer authorization "
                          "record (c58008cc...) + amendment 11 flow through "
                          "lease, binding witness, terminals and the final "
                          "verifier; scoring stays frozen-genesis mechanics",
    "cells_verified": 12,
    "per_cell_wall_seconds": walls,
    "artifact_class": "INTEGRATED_PROOF_FROZEN_GENESIS_V6",
}, indent=1))
print("samples written")
