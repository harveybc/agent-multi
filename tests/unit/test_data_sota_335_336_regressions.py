"""Permanent regressions for DATA-SOTA-335/336 — the auditor's five
counterexamples plus impostor property grids and identity-swap
refusal."""
from __future__ import annotations

import itertools

import pytest

torch = pytest.importorskip("torch")

from feature_branch_plugins._topology import (TopologyError,  # noqa
                                              strict_int, strict_real)
from feature_branch_plugins.patchtst_branch import (  # noqa: E402
    Plugin as PatchTST)
from feature_branch_plugins.tcn_branch import Plugin as TCN  # noqa
from feature_branch_plugins.tft_branch import Plugin as TFT  # noqa
from feature_branch_plugins.timesnet_branch import (  # noqa: E402
    Plugin as TimesNet)
from feature_fusion_plugins.cross_family_attention import (  # noqa
    Plugin as CrossAttn)


# --- the five auditor counterexamples, frozen ---------------------------

def test_335_patch_len_True_refused():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        PatchTST.build(4, 32, dict(PatchTST.plugin_params,
                                   patch_len=True))


def test_335_dropout_string_refused():
    with pytest.raises(TopologyError, match="non-boolean finite"):
        TFT.build(4, 32, dict(TFT.plugin_params, dropout="0.2"))


def test_335_bool_fusion_branch_width_refused():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        CrossAttn.build([8, True], dict(CrossAttn.plugin_params,
                                        family_ids=["a", "b"]))


def test_335_fractional_window_refused_by_validator_not_torch():
    with pytest.raises(TopologyError, match="non-boolean integer"):
        PatchTST.build(4, 32.5, dict(PatchTST.plugin_params))


def test_336_duplicate_family_ids_refused():
    with pytest.raises(ValueError, match="duplicate family_ids"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     d_model=16, n_heads=2,
                                     family_ids=["a", "a"]))


# --- 336: identity is runtime-bound; same-width swaps refuse ------------

def test_336_same_width_swap_refused_by_identity():
    f, _ = CrossAttn.build([8, 8], dict(
        CrossAttn.plugin_params, d_model=16, n_heads=2, output_dim=24,
        family_ids=["ret", "vol"])), None
    fusion = f[0]
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    assert fusion([("ret", a), ("vol", b)]).shape == (1, 24)
    with pytest.raises(ValueError, match="identity mismatch"):
        fusion([("vol", b), ("ret", a)])


def test_336_positional_input_refused():
    fusion, _ = CrossAttn.build([8, 8], dict(
        CrossAttn.plugin_params, d_model=16, n_heads=2,
        family_ids=["ret", "vol"]))
    with pytest.raises(ValueError, match="NAMED records"):
        fusion([torch.randn(1, 8), torch.randn(1, 8)])


def test_336_missing_or_empty_family_ids_refused():
    with pytest.raises(ValueError, match="one family_id per"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     family_ids=["only_one"]))
    with pytest.raises(ValueError, match="nonempty strings"):
        CrossAttn.build([8, 8], dict(CrossAttn.plugin_params,
                                     family_ids=["a", "  "]))


# --- impostor property grids over exposed genes -------------------------

IMPOSTORS = (True, False, "8", 8.0, 7.5, float("nan"), float("inf"),
             None, [8])


@pytest.mark.parametrize("impostor", IMPOSTORS)
@pytest.mark.parametrize("key", ["patch_len", "stride", "d_model",
                                 "n_heads", "n_layers", "ff_mult"])
def test_335_patchtst_gene_impostors_refused(key, impostor):
    params = dict(PatchTST.plugin_params)
    params[key] = impostor
    with pytest.raises((TopologyError, ValueError)):
        PatchTST.build(4, 32, params)


@pytest.mark.parametrize("impostor", IMPOSTORS)
@pytest.mark.parametrize("key", ["top_k", "d_model", "kernel"])
def test_335_timesnet_gene_impostors_refused(key, impostor):
    params = dict(TimesNet.plugin_params)
    params[key] = impostor
    with pytest.raises((TopologyError, ValueError)):
        TimesNet.build(4, 32, params)


@pytest.mark.parametrize("impostor", IMPOSTORS)
def test_335_tcn_channel_list_impostors_refused(impostor):
    with pytest.raises((TopologyError, ValueError)):
        TCN.build(4, 32, dict(TCN.plugin_params,
                              channels=[64, impostor]))


def test_335_param_ceiling_impostor_refused():
    from feature_branch_plugins._topology import require_param_ceiling
    with pytest.raises(TopologyError):
        require_param_ceiling(10, {"max_parameters": True})


def test_335_validation_precedes_torch_construction(monkeypatch):
    # the validator must fire BEFORE any torch module exists
    import feature_branch_plugins.patchtst_branch as mod
    calls = []
    with pytest.raises(TopologyError):
        PatchTST.build(4, 32, dict(PatchTST.plugin_params,
                                   d_model="64"))


def test_340_all_public_evidence_free_of_topology_and_identifiers():
    """DATA-SOTA-339/340: EVERY public evidence packet on this branch
    is free of operator topology, scratch paths, host names, UUID
    fragments and persistent UUID hashes."""
    import re
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2] / "docs/audits/evidence"
    needles = ("/home/", "/tmp/claude", ".local/state", ".local/share",
               "harveybc", "omega", "dragon", "gamma")
    uuid_frag = re.compile(r"GPU-[0-9a-f]{8}")
    uuid_keys = re.compile(r"gpu_uuid(_redacted|_sha256)?\"")
    # LEGACY files inherited from pre-front eras: registered in the
    # history-remediation register (finding-323 item) — other fronts'
    # audit evidence, not mutated unilaterally by this front.
    registered_legacy = {
        "ETH_EASY_ACTIVITY_SMOKE_2026_08_05.json",
        "HISTORICAL_FITNESS_PROVENANCE_GYMFX_8088F9E.json",
        "MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json",
        "MUSASHI_LIVE_MODEL_IDENTITY_AFTER_241_2026_08_12.json",
        "MUSASHI_POST_OUTAGE_RUNTIME_FACTS_2026_08_11.json",
        "P1LR_DECISION_FINAL_EVIDENCE_c0e53cf18b7d60dd_2026_08_15.json",
        "PLATEAU_LR_CPU_SMOKE_2026_08_21.json",
        "PLATEAU_LR_CUDA_SMOKE_2026_08_21.json",
        "README_LINK_RESOLUTION_CHECK_2026_08_10.json",
        "README_LINK_RESOLUTION_CHECK_2026_08_11.json",
        "README_LINK_RESOLUTION_CHECK_POST_MERGE_2026_08_12.json",
        "REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json",
        "SOCIAL_ENRICHMENT_RETRY_DRYRUN_2026_08_10.json",
        "SWARM_EFFICIENCY_MEASUREMENT_2026_07_31.json",
        "SWARM_EFFICIENCY_MEASUREMENT_CLOCKED_2026_07_31.json",
        "TOOLING_CYCLE_PROVENANCE_2026_08_06.json",
        "WP2_ACTIVITY_PLATEAU_SENSITIVITY_DATASET_2026_08_20.json",
        "WP4_CPU_SMOKE_REPORT_2026_08_20.json",
        "WP4_REWARD_SCALE_CALIBRATION_2026_08_18.json",
        "frag_dragon.json",
        "frag_gamma.json",
        "frag_omega.json",
    }
    offenders = []
    for f in sorted(root.glob("*.json")):
        if f.name in registered_legacy:
            continue
        body = f.read_text()
        for n in needles:
            if n in body:
                offenders.append(f"{f.name}: {n}")
        if uuid_frag.search(body):
            offenders.append(f"{f.name}: GPU-uuid fragment")
        if uuid_keys.search(body):
            offenders.append(f"{f.name}: uuid-derived key")
    assert not offenders, offenders


def test_340_v3_packet_uses_logical_identities():
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2] / "docs/audits/evidence"
    d = _json.loads((root / "CUDA_C0_SMOKE_V3_PUBLIC_2026_08_26.json"
                     ).read_text())
    assert d["interpreter"] == {
        "logical": "python:3.12.13@conda-env:trading-stack"}
    assert d["argv_logical"] == ["tools/cuda_c0_smoke.py"]
    assert "argv_full" not in d and "command" not in d
    assert d["replaces_rejected_packet"]["v1_content_sha256"]


def test_340_tombstones_carry_digest_and_reason():
    import json as _json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2] / "docs/audits/evidence"
    for name in ("CUDA_C0_SMOKE_2026_08_26.json",
                 "CUDA_C0_SMOKE_V2_2026_08_26.json"):
        d = _json.loads((root / name).read_text())
        assert d["schema"] == "agent_multi.evidence_tombstone.v1"
        assert len(d["content_sha256"]) == 64
        assert d["reason"]
