#!/usr/bin/env python3
"""P3 — final CPU routing evaluation under the VALIDATED neutral probe
protocol (final probe order; predeclaration committed at 363b61da).

Routes are NOT retrained: the existing R5 route/solo training
identities are reused verbatim (their encoders load by digest-bearing
run dirs). Adds the RANDOM strong-architecture encoder as the neutral
floor. Every encoder is probed through `common_probe_surface_v2`
(adapter-train/val split with purge, early stop + best-state restore,
three fixed seeds, instability refusal), then P2 skill and the
predeclared per-family ranking are applied. Raw losses are preserved.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import sha256_file  # noqa: E402
from agent_plugins.objective_routing import (  # noqa: E402
    ProbeRefusal, common_probe_surface_v2, normalized_skill,
    select_routes, split_adapter_train_val)

PREDECLARATION = (REPO / "docs/audits/evidence/"
                  "FINAL_PROBE_PROTOCOL_PREDECLARATION_2026_08_28.json")
DESIGN_PATH = (REPO / "docs/audits/evidence/"
               "OBJECTIVE_ROUTING_COMMON_PROBE_DESIGN_2026_08_27.json")
LONG = {"reconstruction": "masked_patch_reconstruction",
        "quantile": "multi_horizon_quantile",
        "contrastive": "hierarchical_contrastive",
        "volatility": "volatility", "barrier": "barrier_hit"}
FAMILIES = ("returns_momentum", "trend_level",
            "volatility_distribution", "oscillators", "volume_flow")
PREDICTIVE = ("quantile", "volatility", "barrier")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--routes-workdir", required=True,
                        help="the R5 workdir holding the EXISTING "
                             "route/solo run dirs (identities reused)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import numpy as np
    import torch

    from agent_plugins.branch_pretraining import (
        barrier_hit_labels, build_step_index, collect_preprocessed_windows,
        five_way_split, forward_log_return_targets, load_fit_slice,
        realized_volatility_targets, sample_span_mask, validate_contract)
    from agent_plugins.component_config import deep_merge_strict
    from app.plugin_loader import load_plugin
    from tools.pretrain_branches import resolve_data_path

    predeclaration = json.loads(PREDECLARATION.read_text())
    fit_protocol = {
        "adapter_seeds": predeclaration["adapter_fitting_P1"][
            "adapter_seeds"],
        "max_steps": predeclaration["adapter_fitting_P1"]["max_steps"],
        "min_steps": predeclaration["adapter_fitting_P1"]["min_steps"],
        "validation_cadence_steps": predeclaration[
            "adapter_fitting_P1"]["validation_cadence_steps"],
        "patience_steps": predeclaration["adapter_fitting_P1"][
            "patience_steps"],
        "minimum_improvement_fraction": 0.01,
        "lr": predeclaration["adapter_fitting_P1"]["optimizer"]["lr"],
        "batch_size": predeclaration["adapter_fitting_P1"][
            "optimizer"]["batch_size"],
        "projection_dim": 32}
    design = json.loads(DESIGN_PATH.read_text())
    workdir = Path(args.routes_workdir)
    contract = json.loads((workdir / "contract_full5_control.json"
                           ).read_text())
    parsed = validate_contract(contract)
    data_path, _ = resolve_data_path()
    df, columns, close_col = load_fit_slice(data_path, contract)
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             parsed["window_stride"],
                             parsed["max_horizon_all_objectives"],
                             design["budget"]["max_windows"])
    blocks, _purged = five_way_split(
        steps, design["partitions_five_way"]["fractions"],
        parsed["max_horizon_all_objectives"])
    source_config_path = Path(
        contract["observation_pipeline"]["source_config"])
    if not source_config_path.is_absolute():
        source_config_path = REPO / source_config_path
    env_config = json.loads(source_config_path.read_text())
    all_windows = collect_preprocessed_windows(df, contract, env_config,
                                               steps)
    step_pos = {t: i for i, t in enumerate(steps)}
    fit_idx = np.array([step_pos[t] for t in blocks["probe_fit"]])
    score_idx = np.array([step_pos[t] for t in blocks["probe_score"]])
    train_pos, val_pos = split_adapter_train_val(
        np.arange(len(fit_idx)),
        parsed["max_horizon_all_objectives"])
    close = df[close_col].to_numpy()
    q_spec = contract["objectives"]["multi_horizon_quantile"]
    v_spec = contract["objectives"]["volatility"]
    b_spec = contract["objectives"]["barrier_hit"]
    quantiles = list(q_spec["quantiles"])
    q_all = torch.tensor(forward_log_return_targets(
        close, steps, q_spec["horizons"]))
    v_all = torch.tensor(realized_volatility_targets(
        close, steps, v_spec["horizons"], float(v_spec["epsilon"]),
        None))
    ohlc = b_spec["ohlc_columns"]
    b_all = torch.tensor(barrier_hit_labels(
        df[ohlc["open"]].to_numpy(), df[ohlc["high"]].to_numpy(),
        df[ohlc["low"]].to_numpy(), df[ohlc["close"]].to_numpy(),
        steps, b_spec["horizons"],
        int(b_spec["barrier_scale"]["lookback"]),
        float(b_spec["upper_mult"]), float(b_spec["lower_mult"]),
        float(b_spec["barrier_scale"]["epsilon"])))
    window = parsed["window_size"]
    rec_spec = contract["objectives"]["masked_patch_reconstruction"]
    mask_fit = sample_span_mask(len(fit_idx), window,
                                float(rec_spec["mask_ratio"]),
                                int(rec_spec["mask_span"]),
                                torch.Generator().manual_seed(1235))
    mask_score = sample_span_mask(len(score_idx), window,
                                  float(rec_spec["mask_ratio"]),
                                  int(rec_spec["mask_span"]),
                                  torch.Generator().manual_seed(1236))
    steps_array = np.asarray(steps)
    positions_fit = torch.tensor(steps_array[fit_idx])
    positions_score = torch.tensor(steps_array[score_idx])

    def build_encoder(family):
        branch = next(b for b in contract["branches"]
                      if b["name"] == family)
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch.get("params") or {},
                                   path="branch.params")
        ch_idx = [columns.index(f) for f in branch["features"]]
        encoder, _dim = plugin_class.build(len(ch_idx), window, params)
        return encoder, ch_idx

    def probe(family, state_path=None, floor_mode=False,
              only_tasks=None):
        torch.manual_seed(0)  # random floor: fixed construction seed
        encoder, ch_idx = build_encoder(family)
        if state_path is not None:
            encoder.load_state_dict(torch.load(state_path,
                                               weights_only=True))
        encoder.eval()
        fam = torch.tensor(all_windows[:, :, ch_idx].copy())
        w_fit, w_score = fam[fit_idx], fam[score_idx]
        with torch.no_grad():
            bundle = dict(
                embeddings_fit=encoder(w_fit),
                embeddings_score=encoder(w_score),
                masked_embeddings_fit=encoder(
                    w_fit.masked_fill(mask_fit.unsqueeze(-1), 0.0)),
                masked_embeddings_score=encoder(
                    w_score.masked_fill(mask_score.unsqueeze(-1), 0.0)))
        return common_probe_surface_v2(
            **bundle, windows_fit=w_fit, windows_score=w_score,
            mask_fit=mask_fit, mask_score=mask_score,
            quantile_targets_fit=q_all[fit_idx],
            quantile_targets_score=q_all[score_idx],
            quantile_quantiles=quantiles,
            volatility_targets_fit=v_all[fit_idx],
            volatility_targets_score=v_all[score_idx],
            barrier_labels_fit=b_all[fit_idx],
            barrier_labels_score=b_all[score_idx],
            positions_fit=positions_fit,
            positions_score=positions_score,
            contrastive_exclusion=12, contrastive_temperature=0.2,
            adapter_train_pos=train_pos, adapter_val_pos=val_pos,
            protocol=fit_protocol, floor_mode=floor_mode,
            only_tasks=only_tasks)

    pruned = design["arms_per_family"]["evidence_pruned"]
    pruned_tag = {family: "pruned_" + "_".join(sorted(objectives))
                  for family, objectives in pruned.items()}
    report = {"schema": "agent_multi.final_probe_screen.v1",
              "order": "P3",
              "predeclaration_sha256": sha256_file(PREDECLARATION),
              "route_identities": "REUSED from the R5 run dirs "
                                  "verbatim (no retraining)",
              "families": {}, "verdicts": {}, "selected": {}}
    for family in FAMILIES:
        rows = {}
        rows["random_floor"] = probe(family, None, floor_mode=True)
        for short in LONG:
            # predeclaration: the solo reference is the specialist's
            # OWN-task ceiling only — foreign probes are out of scope.
            # Addendum v2: a ceiling that cannot fit its own task makes
            # that probe DIAGNOSTIC_INVALID, never kills the screen.
            try:
                rows[f"solo_{short}"] = probe(
                    family, workdir / f"solo_{short}"
                    / f"branch_{family}_encoder.pt",
                    only_tasks=[short])
            except ProbeRefusal as refusal:
                rows[f"solo_{short}"] = {"probes": {short: {
                    "probe_score_median": None,
                    "ceiling_diagnostic_invalid": str(refusal)[:120]}}}
        arm_dirs = {"full5_control": "full5_control",
                    "predictive3": "predictive3",
                    "self_supervised2": "self_supervised2",
                    "evidence_pruned": pruned_tag[family]}
        for arm, tag in arm_dirs.items():
            try:
                rows[arm] = probe(family, workdir / tag
                                  / f"branch_{family}_encoder.pt")
            except ProbeRefusal as refusal:
                rows[arm] = {"ROUTE_REFUSED": str(refusal)}
        # P2 skill per arm
        floor_probes = rows["random_floor"]["probes"]
        # DATA-SOTA-375: a marginal/unstable floor can NEVER anchor
        # skill — the task becomes DIAGNOSTIC_INVALID; provenance kept
        random_scores = {task: (None if floor_probes[task].get(
            "floor_fit_marginal") or floor_probes[task].get(
            "floor_diagnostic_invalid")
            else floor_probes[task]["probe_score_median"])
            for task in LONG}
        solo_scores = {task: rows[f"solo_{task}"]["probes"][task][
            "probe_score_median"] for task in LONG}
        floor_invalid = {task for task in LONG
                         if random_scores[task] is None
                         or solo_scores.get(task) is None}
        arm_facts = {}
        for arm in arm_dirs:
            facts = rows[arm]
            if "ROUTE_REFUSED" in facts:
                arm_facts[arm] = facts
                continue
            skills = {}
            invalid = {}
            for task in LONG:
                if task in floor_invalid:
                    invalid[task] = ("DIAGNOSTIC_INVALID: unreliable "
                                     "random floor (addendum)")
                    continue
                route_score = facts["probes"][task][
                    "probe_score_median"]
                skill, reason = normalized_skill(
                    random_scores[task], route_score,
                    solo_scores[task])
                if reason:
                    invalid[task] = reason
                else:
                    skills[task] = skill
            arm_facts[arm] = {
                "raw_probe_medians": {t: facts["probes"][t][
                    "probe_score_median"] for t in LONG},
                "skills": skills, "diagnostic_invalid": invalid,
                "dispersions": {t: facts["probes"][t]["dispersion"]
                                for t in LONG}}
        report["families"][family] = {
            "random_floor_provenance": floor_probes,
            "random_floor_raw": random_scores,
            "solo_ceiling_raw": solo_scores,
            "arms": arm_facts}
    # DATA-SOTA-374/376: selection authority is the pure function —
    # refused arms can never be selected, eligibility needs all three
    # predictive skills valid
    authority = select_routes(report["families"])
    report["verdicts"] = authority["verdicts"]
    report["selected"] = authority["selected"]
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdicts"], indent=1))
    print(json.dumps({f: s for f, s in report["selected"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
