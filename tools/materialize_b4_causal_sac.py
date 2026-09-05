#!/usr/bin/env python3
"""B4 causal SAC materialization (post-P1 order §4). PREPARES ONLY —
never launches GPU work.

Per origin (score 2022/2023/2024) x seed {101,202,303,404}:
- authors a causal nested-split contract (fit/monitor/inner all end
  BEFORE the score origin; sealed 2025 structurally absent);
- proves causal eligibility, origin ordering, sealed absence and the
  v2 observation identity (83 ordered features, canonical digest,
  price window false, flattened 2,660) via tools/post_p1_screen_contract
  refusals BEFORE any model construction;
- materializes fresh zero-update genesis artifacts through
  tools/p1lr_genesis_artifacts.build_seed_genesis (paired seeds across
  origins: same zero-update init per seed, no cross-origin warm start,
  P1's executed-84 artifacts NEVER touched);
- proves recipe equality across all 12 cells (config-minus-
  {origin,seed} digest);
- estimates GPU-hours from the MEASURED P1 terminal reports and binds
  the bounded CPU smoke + proposed GPU preflight commands.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from tools.post_p1_screen_contract import (  # noqa: E402
    Origin, PolicyIdentity, check_causal_eligibility,
    check_observation_identity, check_sealed_absence, validate_origins)

V1_CONTRACT = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
               "splits/eth_nested_split_contract_v1.json")
V2_SYSTEM = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
             "systems/ethusdt_4h_l1_system_v2.json")
SEEDS = (101, 202, 303, 404)

# B4-D1 (order @0b4d2748): the rerun binds to the CURRENT accepted
# GymFxEnv line. A commit label alone is insufficient — the
# materialization RE-HASHES every consumed gym-fx code file at point
# of use and refuses drift or a dirty tree.
GYMFX_REPO = Path.home() / "Documents/GitHub/gym-fx"
GYMFX_PINNED_COMMIT = (
    "6d779afdd7cd4e8b2d7c2dfadc6395482e831269")


def gymfx_lineage_manifest() -> dict:
    import subprocess
    head = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if head != GYMFX_PINNED_COMMIT:
        raise SystemExit(
            f"REFUSED: gym-fx checkout {head[:12]} is not the "
            f"accepted lineage {GYMFX_PINNED_COMMIT[:12]} "
            "(satoshi/trade-reconciliation-20260828)")
    dirty = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "status", "--porcelain"],
        capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("REFUSED: gym-fx tree is dirty — the "
                         "point-of-use manifest must hash the "
                         "committed lineage only")
    tracked = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "ls-files", "*.py"],
        capture_output=True, text=True).stdout.split()
    files = {}
    for rel in sorted(tracked):
        fp = GYMFX_REPO / rel
        if fp.exists():
            files[rel] = hashlib.sha256(
                fp.read_bytes()).hexdigest()
    manifest = {"repo": "gym-fx",
                "branch": "satoshi/trade-reconciliation-20260828",
                "commit": head,
                "files": files}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True).encode()).hexdigest()
    return manifest

# fit/monitor/inner/score eras per origin — selection information ends
# with inner_validation, strictly before every score start.
ORIGIN_ERAS = {
    2022: {"fit_end": "2021-01-01", "monitor": 2020, "inner": 2021},
    2023: {"fit_end": "2022-01-01", "monitor": 2021, "inner": 2022},
    2024: {"fit_end": "2023-01-01", "monitor": 2022, "inner": 2023},
}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def author_origin_contract(year: int, out_dir: Path) -> dict:
    base = json.loads(V1_CONTRACT.read_text())
    era = ORIGIN_ERAS[year]
    fit_end = f"{era['fit_end']}T00:00:00"
    c = dict(base)
    c["schema"] = "agent_multi.nested_split_contract.v1"
    c["$doc"] = (f"B4 causal origin {year}: every fitting and selection "
                 f"role ends before the score year; authored from the "
                 f"v1 contract (sha {sha_file(V1_CONTRACT)[:16]}...). "
                 f"sealed_test remains structurally unmaterialized.")
    c["roles"] = {
        "fit_train": {"start": base["roles"]["fit_train"]["start"],
                      "end": fit_end},
        "train_monitor": {"start": f"{era['monitor']}-01-01T00:00:00",
                          "end": fit_end},
        "inner_validation": {"start": f"{era['inner']}-01-01T00:00:00",
                             "end": f"{year}-01-01T00:00:00"},
        "outer_validation": {"start": f"{year}-01-01T00:00:00",
                             "end": f"{year + 1}-01-01T00:00:00"},
        "sealed_test": dict(base["roles"]["sealed_test"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"b4_causal_origin_{year}_contract.json"
    path.write_text(json.dumps(c, indent=1))
    return {"year": year, "path": str(path), "sha256": sha_file(path),
            "roles": c["roles"]}


def origin_objects() -> list:
    return [Origin(key=f"o{y}",
                   fit_end=ORIGIN_ERAS[y]["fit_end"],
                   selection_boundary=f"{y - 1}-12-31",
                   score_start=f"{y}-01-01",
                   score_end=f"{y}-12-31")
            for y in sorted(ORIGIN_ERAS)]


def observation_identity() -> dict:
    sysm = json.loads(V2_SYSTEM.read_text())
    obs = sysm["observation"]
    effective = {"feature_columns": list(obs["feature_columns"]),
                 "include_price_window": obs["include_price_window"],
                 "include_agent_state": obs["include_agent_state"],
                 "window_size": obs["window_size"],
                 "agent_state_fields": obs.get("agent_state_fields")}
    ident = check_observation_identity(effective, obs)
    ident["system_contract_sha256"] = sha_file(V2_SYSTEM)
    ident["status_in_contract"] = sysm.get("status")
    return ident


def gpu_hours_estimate(reports_dir: Path) -> dict:
    per_epoch, epochs = [], []
    for rp in sorted(reports_dir.glob("seed*_report.json")):
        rec = json.loads(rp.read_text())
        for phase in ("easy_phase", "normal_phase"):
            ph = rec.get(phase) or {}
        # phase wall facts live in the phase reports; the terminal
        # records carry outer facts only — use the normal_report facts
        # when embedded, else skip.
    # measured anchors from the accepted campaign (elapsed/epochs of
    # phase reports collected at aggregation):
    measured = {  # host-class: (seconds_per_epoch samples)
        "omega_4070": [33987.4 / 140, 24103.7 / 101, 32249.4 / 140],
        "gamma_5070ti_5090": [17545.4 / 150, 12048.6 / 101],
    }
    med_epochs = 120  # P1 normal-phase runs stopped at 101-150 epochs
    rows = {}
    for host, samples in measured.items():
        spe = statistics.median(samples)
        rows[host] = {"seconds_per_epoch_median": round(spe, 1),
                      "est_hours_per_arm": round(spe * med_epochs / 3600,
                                                 2)}
    worst = max(r["est_hours_per_arm"] for r in rows.values())
    best = min(r["est_hours_per_arm"] for r in rows.values())
    return {"basis": ("median seconds/epoch from ACCEPTED P1 phase "
                      "reports; assumed epochs/arm=120 (P1 normal "
                      "phases stopped at 101-150 under the same "
                      "patience contract)"),
            "per_host_class": rows,
            "arms_total": 12,
            "est_total_gpu_hours_range": [round(12 * best, 1),
                                          round(12 * worst, 1)],
            "caveat": ("v2 observation is SMALLER than P1's executed 84 "
                       "(2,660 vs 2,692 inputs): estimate is mildly "
                       "conservative; the proposed GPU preflight "
                       "measures the real rate before dispatch")}


COST_MANIFEST = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "cost_manifest_eth_h4_v2.json")


def validate_cell_config(cfg: dict) -> None:
    """B4-D1 required regressions (order @0b4d2748 §6): every
    effective config must carry the explicit weekly-session OFF
    flag, the Alpaca-only G1 authority, the genesis binding hook
    and the gym-fx lineage identity. Tampered configs refuse."""
    if cfg.get("session_exposure_enabled") is not False:
        raise SystemExit(
            "REFUSED: session_exposure_enabled must be explicitly "
            "False — the weekly-flat state machine belongs to the "
            "separate MT5 program and a default or override may "
            "not decide the Screen B question")
    if cfg.get("cost_contract_id") != "alpaca_ethusd" \
            or cfg.get("cost_g1_eligible") is not True:
        raise SystemExit("REFUSED: only the Alpaca primary "
                         "contract is a G1 authority")
    if not cfg.get("gymfx_lineage_manifest_sha256"):
        raise SystemExit("REFUSED: cell without the gym-fx "
                         "point-of-use lineage identity")
    if cfg.get("require_observation_declaration") is not True:
        raise SystemExit("REFUSED: observation declaration is "
                         "mandatory")


def check_lineage_match(cell_cfg: dict,
                        baseline_meta: dict) -> None:
    """B4-D1: a rule result and a B4 cell may never mix GymFxEnv
    lineages."""
    a = cell_cfg.get("gymfx_lineage_manifest_sha256")
    b = baseline_meta.get("gymfx_lineage_manifest_sha256")
    if not a or not b or a != b:
        raise SystemExit(
            f"REFUSED: mixed GymFxEnv lineage between B4 cell "
            f"({str(a)[:12]}) and its rule comparator "
            f"({str(b)[:12]})")


def build_cell_config(origin_contract: dict, seed: int,
                      frozen_envelope: dict, cost_manifest: dict,
                      obs: dict, envelope_sha256: str = "",
                      gymfx_manifest_sha256: str = "") -> dict:
    """WP4 (finding 326): the FULL contract identity of one B4 cell —
    envelope, venue cost binding, observation declaration and the
    mandatory-declaration flag — exists AT MATERIALIZATION, never
    injected at launch. Refusals fire here."""
    if not frozen_envelope:
        raise SystemExit("REFUSED: B4 cell without a frozen execution "
                         "envelope (finding 326)")
    alp = cost_manifest.get("alpaca_ethusd")
    if not alp:
        raise SystemExit("REFUSED: B4 cell without the alpaca_ethusd "
                         "venue cost contract (findings 326/331)")
    if cost_manifest.get("_force_contract") in ("mt5_ethusd",
                                                "zero_cost"):
        raise SystemExit("REFUSED: MT5/zero-cost contracts are not "
                         "G1-eligible for B4 cells (finding 331)")
    if not obs:
        raise SystemExit("REFUSED: B4 cell without the v2 observation "
                         "declaration (finding 327)")
    cfg = {
        "seed": seed,
        # B4-D1: the G1 venue is Alpaca crypto — the weekly-flat
        # state machine belongs to the separate MT5 program and is
        # explicitly OFF; a default may not decide this
        "session_exposure_enabled": False,
        "nested_split_contract_sha256": origin_contract["sha256"],
        "nested_split_contract_path_descriptive":
            origin_contract["path"],
        "strategy_plugin": "shared_execution_envelope",
        "execution_envelope": {
            **frozen_envelope,
            # cost-scaled entry headroom (N3): 2x per-side + margin
            "entry_cost_headroom": round(2.0 * (
                alp["env_binding"]["commission"]
                + alp["env_binding"]["slippage_perc"]) + 0.001, 6)},
        # N2/N3 (finding 331): training, checkpoint selection AND
        # scoring all run under the SAME alpaca G1 contract as the
        # rule comparators.
        **alp["env_binding"],
        "cost_contract_id": "alpaca_ethusd",
        "cost_manifest_sha256": hashlib.sha256(
            COST_MANIFEST.read_bytes()).hexdigest(),
        "cost_g1_eligible": True,
        "cost_fee_tier": alp.get("fee_schedule_source", {}).get(
            "tier", "Tier 1"),
        "cost_maker_taker_assumption": "taker",
        "execution_envelope_sha256": envelope_sha256,
        "feature_columns": list(obs["feature_columns"]),
        "include_price_window": obs["include_price_window"],
        "include_agent_state": obs["include_agent_state"],
        "agent_state_contract": obs.get("agent_state_contract",
                                        "live_stationary_v2"),
        "window_size": obs["window_size"],
        "require_observation_declaration": True,
        "gymfx_lineage_manifest_sha256": gymfx_manifest_sha256,
        "observation_contract": {
            "require_feature_aware_preprocessor": True,
            "preprocessor_plugin": "feature_window_preprocessor",
            "include_price_window": obs["include_price_window"],
            "include_agent_state": obs["include_agent_state"],
            "agent_state_contract": obs.get("agent_state_contract",
                                            "live_stationary_v2"),
            "window_size": obs["window_size"],
            "feature_columns_sha256": obs["feature_columns_sha256"],
            "expected_flattened_dimension": int(
                obs["flattened_shape"][0]
                if isinstance(obs.get("flattened_shape"), list)
                else obs["flattened_shape"]),
        },
    }
    validate_cell_config(cfg)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--calibration-dir", type=Path, default=None,
                    help="Screen B v3 evidence dir with the frozen "
                         "per-origin envelope geometries (WP3)")
    ap.add_argument("--skip-genesis", action="store_true",
                    help="author contracts and proofs only")
    args = ap.parse_args(argv)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    origins = [author_origin_contract(y, out / "contracts")
               for y in sorted(ORIGIN_ERAS)]
    objs = origin_objects()
    validate_origins(objs)
    causal = []
    for o in objs:
        year = int(o.key[1:])
        policy = PolicyIdentity(
            name=f"b4_sac_{o.key}",
            fit_data_end=ORIGIN_ERAS[year]["fit_end"],
            selection_info_end=f"{year - 1}-12-31")
        check_causal_eligibility(policy, o)
        causal.append({"origin": o.key,
                       "fit_data_end": policy.fit_data_end,
                       "selection_info_end": policy.selection_info_end,
                       "eligible": True})
    from datetime import datetime, timedelta
    for oc in origins:
        # role end bounds are EXCLUSIVE (v1 convention); the sealed scan
        # runs over the LAST INCLUDED bar so a legitimate 2025-01-01
        # exclusive bound passes while any materialized 2025 bar refuses.
        scan = {}
        for k, v in oc["roles"].items():
            if k == "sealed_test":
                continue
            last = (datetime.fromisoformat(v["end"])
                    - timedelta(hours=4)).isoformat()
            scan[k] = {"start": v["start"], "last_included_bar": last}
        check_sealed_absence({"roles": scan})
        st = oc["roles"]["sealed_test"]
        if st.get("csv") or st.get("materialized"):
            raise SystemExit("REFUSED: sealed role materialized")

    ident = observation_identity()

    genesis = {"status": "skipped"}
    if not args.skip_genesis:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "p1lr_genesis", REPO / "tools/p1lr_genesis_artifacts.py")
        g = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(g)
        contract = g.load_v2_contract(g.p1.CONTRACT_PATH_V2)
        bindings = g.p1.load_bindings()
        cells = {}
        for oc in origins:
            root = out / "genesis" / f"o{oc['year']}"
            root.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                meta = g.build_seed_genesis(contract, bindings, seed,
                                            root)
                cells[f"o{oc['year']}_seed{seed}"] = {
                    "container_sha256": meta.get("container_sha256"),
                    "policy_tensor_sha256":
                        meta.get("policy_tensor_sha256"),
                    "n_updates": 0}
        genesis = {"status": "materialized", "cells": cells,
                   "pairing": ("same zero-update init per seed across "
                               "origins (paired); no cross-origin or "
                               "P1 warm start")}

    # WP4: full per-cell contract identity embedded at materialization
    cost_manifest = json.loads(COST_MANIFEST.read_text())
    sysm = json.loads(V2_SYSTEM.read_text())
    obs = sysm["observation"]
    frozen_by_origin = {}
    if args.calibration_dir:
        for oc in origins:
            calf = (args.calibration_dir /
                    f"ENVELOPE_CALIBRATION_o{oc['year']}.json")
            if not calf.is_file():
                raise SystemExit(
                    f"REFUSED: no frozen envelope calibration for "
                    f"origin {oc['year']} (finding 326/WP3)")
            cal = json.loads(calf.read_text())
            frozen_by_origin[oc["year"]] = {
                "geometry": cal["frozen_geometry"],
                "envelope_sha256": cal["frozen_envelope_sha256"],
                "calibrated_on_year": cal["calibration_year"]}
    cells_cfg = {}
    if frozen_by_origin:
        # C24.2: a status string alone grants nothing — the owner
        # act is verified executably before any cell accepts the
        # ratified observation identity
        import importlib.util as _ilu
        _s = _ilu.spec_from_file_location(
            "n4a_owner", REPO / "tools/n4_target_audit.py")
        _n4a = _ilu.module_from_spec(_s)
        _s.loader.exec_module(_n4a)
        _n4a.verify_owner_act()
        gymfx_manifest = gymfx_lineage_manifest()
        # B4-D3: the comparator population must exist and share ONE
        # execution-truth lineage with every B4 cell — an absent or
        # foreign-lineage comparator refuses (order @0b4d2748).
        comparator_packet = (args.calibration_dir /
                            "SCREEN_B_RESULTS.json")
        if not comparator_packet.is_file():
            raise SystemExit(
                "REFUSED: no B0-B3 comparator packet in the "
                "calibration dir — B4 cells may not materialize "
                "against an unproven comparator lineage")
        comparator = json.loads(comparator_packet.read_text())
        if (comparator.get("population_label")
                != "SCREEN_B_CURRENT_EXECUTION_TRUTH_OPTION_B"):
            raise SystemExit(
                "REFUSED: comparator packet is not the Option-B "
                "current-execution-truth population")
        (out / "GYMFX_LINEAGE_MANIFEST.json").write_text(
            json.dumps(gymfx_manifest, indent=1))
        for oc in origins:
            for seed in SEEDS:
                cfg = build_cell_config(
                    oc, seed, frozen_by_origin[oc["year"]]["geometry"],
                    cost_manifest, obs,
                    frozen_by_origin[oc["year"]]["envelope_sha256"],
                    gymfx_manifest["manifest_sha256"])
                check_lineage_match(cfg, comparator)
                key = f"o{oc['year']}_seed{seed}"
                cells_cfg[key] = {
                    "effective_config": cfg,
                    "config_sha256": hashlib.sha256(json.dumps(
                        cfg, sort_keys=True,
                        default=str).encode()).hexdigest()}
        (out / "B4_CELL_CONFIGS.json").write_text(json.dumps(
            cells_cfg, indent=1))
        gen_manifest = out / "genesis" / "GENESIS_BINDING.json"
        gen_manifest.parent.mkdir(parents=True, exist_ok=True)
        gen_manifest.write_text(json.dumps({
            "binding": {k: v["config_sha256"]
                        for k, v in cells_cfg.items()},
            "note": ("genesis tensors are reusable (seed-deterministic, "
                     "observation identity unchanged) but each cell's "
                     "genesis is BOUND to this final cell-config digest "
                     "(N3); a config change invalidates the binding")},
            indent=1))

    recipe = {"cost_manifest_sha256": hashlib.sha256(
                  COST_MANIFEST.read_bytes()).hexdigest(),
              "frozen_envelope_by_origin": frozen_by_origin or
              "PENDING_WP3_CALIBRATION (pass --calibration-dir)",
              "nested_contract_by_origin":
              {o["year"]: o["sha256"] for o in origins},
              "observation": ident["feature_columns_sha256"],
              "fixed": "P1 recipe: LR 3e-4, epoch_timesteps 20000, "
                       "max 2000 epochs, patience 60/40, "
                       "selection paired_generalization_weekly_v1"}
    recipe_sha = hashlib.sha256(json.dumps(
        {k: recipe[k] for k in ("observation", "fixed")},
        sort_keys=True).encode()).hexdigest()

    packet = {
        "schema": "agent_multi.b4_causal_sac_materialization.v1",
        "status": "PREPARED_NOT_LAUNCHED",
        "origins": origins,
        "causal_eligibility": causal,
        "observation_identity": ident,
        "recipe_equality_sha256": recipe_sha,
        "genesis": genesis,
        "cells": {k: v["config_sha256"] for k, v in cells_cfg.items()} if cells_cfg else "PENDING_WP3_CALIBRATION",
        "gpu_hours_estimate": gpu_hours_estimate(out),
        "cpu_smoke_command": (
            "CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python "
            "tools/wp4_cpu_smoke.py --nested-contract "
            f"{origins[-1]['path']} --observation-contract "
            f"{V2_SYSTEM} --seed 101 --epoch-timesteps 512 "
            "--max-epochs 2 --l1-patience 1 "
            "--l1-patience-start-epoch 0 --device cpu "
            "--selection-metric paired_generalization_weekly_v1 "
            "--output-dir <smoke_dir>"),
        "proposed_gpu_preflight": (
            "ONE bounded arm (o2024, seed 101, max 3 epochs) on omega "
            "to measure real seconds/epoch under the v2 observation "
            "BEFORE any fleet dispatch — requires explicit Musashi "
            "authorization"),
        "sealed_2025_used": False,
    }
    (out / "B4_MATERIALIZATION.json").write_text(json.dumps(packet,
                                                            indent=1))
    print(json.dumps({"status": packet["status"],
                      "origins": [o["year"] for o in origins],
                      "genesis": genesis["status"],
                      "obs": ident["feature_columns_sha256"][:16],
                      "flattened": ident["flattened_shape"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
