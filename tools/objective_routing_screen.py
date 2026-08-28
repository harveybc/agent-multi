#!/usr/bin/env python3
"""R5 — bounded CPU 20-arm objective-routing screen with the COMMON
five-probe surface (routing order 2026-08-27). Executes EXACTLY the
committed design (`OBJECTIVE_ROUTING_COMMON_PROBE_DESIGN_2026_08_27
.json`): five-way purged partitions, fixed invloss+pcgrad mechanism
(FIXED_MECHANISM_NOT_M2_WINNER), solo references through the SAME
surface bound before ranking, per-family lexicographic verdict with
the 0.02 tie tolerance. Quarantined designs refuse. Monitor, 2022,
outer 2024 and sealed 2025 are structurally unavailable to selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.branch_pretraining import (  # noqa: E402
    sha256_file)
from agent_plugins.objective_routing import (  # noqa: E402
    ProbeRefusal, common_probe_surface)

DESIGN_PATH = (REPO / "docs/audits/evidence/"
               "OBJECTIVE_ROUTING_COMMON_PROBE_DESIGN_2026_08_27.json")
REGISTER_PATH = (REPO / "docs/audits/evidence/"
                 "GENERATION_QUARANTINE_REGISTER.json")
LONG = {"reconstruction": "masked_patch_reconstruction",
        "quantile": "multi_horizon_quantile",
        "contrastive": "hierarchical_contrastive",
        "volatility": "volatility", "barrier": "barrier_hit"}
FAMILIES = ("returns_momentum", "trend_level",
            "volatility_distribution", "oscillators", "volume_flow")


def refuse_quarantined(design_digest: str) -> None:
    if REGISTER_PATH.exists():
        register = json.loads(REGISTER_PATH.read_text())
        entry = (register.get("design_entries") or {}).get(
            design_digest)
        if entry:
            raise SystemExit(f"REFUSED: design QUARANTINED as "
                             f"{entry['class']} (R0)")


def run_runner(contract_file: Path, out_dir: Path, epochs: int,
               max_windows: int) -> dict:
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools/pretrain_branches.py"),
         "--contract", str(contract_file), "--output-dir", str(out_dir),
         "--epochs", str(epochs), "--max-windows", str(max_windows)],
        capture_output=True, text=True, cwd=str(REPO))
    if proc.returncode != 0:
        raise SystemExit(f"runner failed for {contract_file.name}:\n"
                         f"{proc.stderr[-1500:]}")
    return json.loads((out_dir / "pretrain_manifest.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True,
                        help="five-objective base contract")
    parser.add_argument("--workdir", required=True)
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

    design = json.loads(DESIGN_PATH.read_text())
    design_digest = sha256_file(DESIGN_PATH)
    refuse_quarantined(design_digest)
    # R0: if the quarantined M3 v1 content were substituted at the
    # design path, its digest is in the register and refuses above

    base = json.loads(Path(args.contract).read_text())
    base["partition_scheme"] = {
        "scheme": "five_way_probe",
        "fractions": design["partitions_five_way"]["fractions"]}
    base["objective_balancing"] = {"method": "inverse_initial_loss",
                                   "floor": 1e-6}
    base["gradient_combiner"] = {"plugin": "pcgrad"}
    epochs = design["budget"]["epochs"]
    max_windows = design["budget"]["max_windows"]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    def derive(objectives_short, tag):
        variant = json.loads(json.dumps(base))
        variant["objectives"] = {
            LONG[o]: base["objectives"][LONG[o]]
            for o in objectives_short}
        path = workdir / f"contract_{tag}.json"
        path.write_text(json.dumps(variant, indent=1))
        return path

    # runs: 5 solos + universal arms + deduped evidence_pruned sets
    runs: dict[str, Path] = {}
    for short in LONG:
        contract_file = derive([short], f"solo_{short}")
        run_runner(contract_file, workdir / f"solo_{short}", epochs,
                   max_windows)
        runs[f"solo_{short}"] = workdir / f"solo_{short}"
    arm_sets = {"full5_control": design["arms_per_family"][
                    "full5_control"],
                "predictive3": design["arms_per_family"]["predictive3"],
                "self_supervised2": design["arms_per_family"][
                    "self_supervised2"]}
    for tag, objectives in arm_sets.items():
        contract_file = derive(objectives, tag)
        run_runner(contract_file, workdir / tag, epochs, max_windows)
        runs[tag] = workdir / tag
    pruned = design["arms_per_family"]["evidence_pruned"]
    pruned_run_by_set: dict[frozenset, str] = {}
    for family, objectives in pruned.items():
        key = frozenset(objectives)
        if key not in pruned_run_by_set:
            tag = "pruned_" + "_".join(sorted(objectives))
            contract_file = derive(sorted(objectives), tag)
            run_runner(contract_file, workdir / tag, epochs,
                       max_windows)
            runs[tag] = workdir / tag
            pruned_run_by_set[key] = tag

    # ---- probe data (once)
    contract = json.loads((workdir / "contract_full5_control.json"
                           ).read_text())
    parsed = validate_contract(contract)
    data_path, _ = resolve_data_path()
    df, columns, close_col = load_fit_slice(data_path, contract)
    steps = build_step_index(len(df), parsed["warmup_bars"],
                             parsed["window_stride"],
                             parsed["max_horizon_all_objectives"],
                             max_windows)
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
    close = df[close_col].to_numpy()
    q_spec = base["objectives"]["multi_horizon_quantile"]
    v_spec = base["objectives"]["volatility"]
    b_spec = base["objectives"]["barrier_hit"]
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
    protocol = dict(design["common_probe_protocol"]["adapters"])
    window = parsed["window_size"]
    rec_spec = base["objectives"]["masked_patch_reconstruction"]
    mask_fit = sample_span_mask(len(fit_idx), window,
                                float(rec_spec["mask_ratio"]),
                                int(rec_spec["mask_span"]),
                                torch.Generator().manual_seed(
                                    int(protocol["seed"]) + 1))
    mask_score = sample_span_mask(len(score_idx), window,
                                  float(rec_spec["mask_ratio"]),
                                  int(rec_spec["mask_span"]),
                                  torch.Generator().manual_seed(
                                      int(protocol["seed"]) + 2))
    steps_array = np.asarray(steps)
    positions_fit = torch.tensor(steps_array[fit_idx])
    positions_score = torch.tensor(steps_array[score_idx])

    def probe(run_dir: Path, family: str) -> dict:
        branch = next(b for b in contract["branches"]
                      if b["name"] == family)
        plugin_class, _ = load_plugin("feature_branch.plugins",
                                      branch["plugin"])
        params = deep_merge_strict(plugin_class.plugin_params,
                                   branch.get("params") or {},
                                   path="branch.params")
        ch_idx = [columns.index(f) for f in branch["features"]]
        encoder, _dim = plugin_class.build(len(ch_idx), window, params)
        state = torch.load(run_dir / f"branch_{family}_encoder.pt",
                           weights_only=True)
        encoder.load_state_dict(state)
        encoder.eval()
        fam = torch.tensor(all_windows[:, :, ch_idx].copy())
        w_fit, w_score = fam[fit_idx], fam[score_idx]
        with torch.no_grad():
            e_fit = encoder(w_fit)
            e_score = encoder(w_score)
            me_fit = encoder(w_fit.masked_fill(
                mask_fit.unsqueeze(-1), 0.0))
            me_score = encoder(w_score.masked_fill(
                mask_score.unsqueeze(-1), 0.0))
        return common_probe_surface(
            embeddings_fit=e_fit, embeddings_score=e_score,
            masked_embeddings_fit=me_fit,
            masked_embeddings_score=me_score,
            windows_fit=w_fit, windows_score=w_score,
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
            protocol=protocol)

    # solo references FIRST (bound before route ranking)
    references: dict[str, dict[str, float]] = {}
    solo_surfaces: dict[str, dict] = {}
    for short in LONG:
        for family in FAMILIES:
            surface = probe(runs[f"solo_{short}"], family)
            solo_surfaces[f"{short}:{family}"] = surface
            references.setdefault(family, {})[short] = \
                surface["probes"][short]["probe_score"]

    report = {"schema": "agent_multi.objective_routing_screen.v1",
              "order": "R5", "design_sha256": design_digest,
              "mechanism_label": design["mechanism"]["label"],
              "solo_references": references,
              "families": {}, "verdicts": {}}
    import statistics
    for family in FAMILIES:
        family_arms = {}
        for arm in ("full5_control", "predictive3", "self_supervised2",
                    "evidence_pruned"):
            if arm == "evidence_pruned":
                run_dir = runs[pruned_run_by_set[
                    frozenset(pruned[family])]]
                trained = sorted(pruned[family])
            else:
                run_dir = runs[arm]
                trained = sorted(design["arms_per_family"][arm])
            try:
                surface = probe(run_dir, family)
                ratios = {task: round(
                    surface["probes"][task]["probe_score"]
                    / references[family][task], 4)
                    for task in LONG}
                degraded = [task for task, ratio in ratios.items()
                            if ratio > 1.2]
                ordered = sorted(ratios.values())
                family_arms[arm] = {
                    "trained_objectives_reported_not_rewarded":
                        trained,
                    "normalized_probe_ratios": ratios,
                    "degraded_common_probes": degraded,
                    "degraded_count": len(degraded),
                    "worst_ratio": ordered[-1],
                    "median_ratio": round(
                        statistics.median(ordered), 4),
                    "encoder_output_std":
                        surface["encoder_output_std"]}
            except ProbeRefusal as refusal:
                family_arms[arm] = {"ROUTE_REFUSED": str(refusal)}
        report["families"][family] = family_arms
        ranked = sorted(
            [(arm, facts) for arm, facts in family_arms.items()
             if "ROUTE_REFUSED" not in facts],
            key=lambda kv: (kv[1]["degraded_count"],
                            kv[1]["worst_ratio"],
                            kv[1]["median_ratio"]))
        if not ranked:
            report["verdicts"][family] = "ALL_ROUTES_REFUSED"
            continue
        best_arm, best = ranked[0]
        ties = [arm for arm, facts in ranked[1:]
                if facts["degraded_count"] == best["degraded_count"]
                and abs(facts["worst_ratio"] - best["worst_ratio"])
                <= 0.02
                and abs(facts["median_ratio"] - best["median_ratio"])
                <= 0.02]
        if ties:
            report["verdicts"][family] = (
                f"INCONCLUSIVE: {best_arm} ties {ties}")
        elif best["degraded_count"] > 0:
            report["verdicts"][family] = (
                f"NO_ACCEPTABLE_ROUTE: best {best_arm} with "
                f"{best['degraded_count']} degraded common probes")
        else:
            report["verdicts"][family] = (
                f"ACCEPTABLE: {best_arm} (zero degraded common "
                f"probes)")
    Path(args.output).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
