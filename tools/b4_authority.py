#!/usr/bin/env python3
"""B4 authority module (order @61622469, B4-E1..E7).

ONE module owns the Screen B/B4 trust roots: the sealed superseding
design and its append-only amendment chain, the gym-fx point-of-use
lineage, the single economic envelope rule shared by comparator and
B4, the complete-envelope digest, the corrected cost-authority
language, the evidence-complete comparator verifier and the
full-chain verifier every scoring path must call before construction.
The caller cannot choose any trust root: reviewed identities are
carried constants here."""
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs/audits/evidence"
GYMFX_REPO = Path.home() / "Documents/GitHub/gym-fx"

GYMFX_PINNED_COMMIT = (
    "6d779afdd7cd4e8b2d7c2dfadc6395482e831269")

# --- Sealed design + append-only amendment chain (E3) -------------
DESIGN_PATH = (EVIDENCE /
               "B4_SUPERSEDING_DESIGN_V2_OPTION_B_2026_09_05.json")
DESIGN_SHA = ("9155f508afc4b87f345a652070a6727a13373c75877c619f"
              "6110d54e9e678237")
AMENDMENT_PATHS = (
    EVIDENCE / "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_1_2026_09_05.json",
    EVIDENCE / "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_2_2026_09_05.json",
    EVIDENCE / "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_3_2026_09_05.json",
)
AMENDMENT_SHAS = (
    "ae874b68ec896e99aa31a309a29e72f9a278cfeff7b9ffa7e1fdb7c72192c805",
    "81f9815fd7c76ba3f0b9ec476f00d507e30b68ecc9a8d202b7c6c4902daaa21a",
    "f04823b77acab35e6822d55016b2aa7bcc996b7b4528071b263b8f8b885835dd",
)
AMENDMENT_4_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_4_2026_09_05"
                    ".json")

# --- Cost-authority language (E6) ---------------------------------
COST_AUTHORITY = (
    "MUSASHI_REVIEWED_FIXED_EXPERIMENTAL_COST_MODEL — Alpaca G1 "
    "selected by auditor review in the prior Screen B order; the "
    "owner act 399483a1... ratified observation v2 and MT5 build "
    "6140, NOT Alpaca costs")
FORBIDDEN_AUTHORITY_PHRASES = (
    "pending ratification",
    "owner-ratified venue path",
    "owner-ratified alpaca",
    "wp4_cpu_smoke",
)

# --- One economic envelope (E1) -----------------------------------
HEADROOM_MARGIN = 0.006
ENVELOPE_ECONOMIC_KEYS = (
    "envelope_mode", "collision_rule", "sizing_mode", "leverage_cap",
    "entry_cost_headroom")


class B4AuthorityRefusal(SystemExit):
    pass


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_obj(o) -> str:
    return hashlib.sha256(json.dumps(
        o, sort_keys=True, default=str).encode()).hexdigest()


def entry_cost_headroom(cost_binding: dict) -> float:
    """B4-E1: the ONE headroom rule for comparator and B4 —
    2x per-side cost + the fixed 0.006 decision-to-fill margin.
    Never averaged, parameterized or tuned."""
    per_side = (float(cost_binding["commission"])
                + float(cost_binding["slippage_perc"]))
    return round(2.0 * per_side + HEADROOM_MARGIN, 6)


def complete_execution_envelope(geometry: dict,
                                cost_binding: dict) -> dict:
    """The complete effective envelope: geometry + the one headroom
    rule. Every field that can alter positions or returns."""
    env = dict(geometry)
    env["entry_cost_headroom"] = entry_cost_headroom(cost_binding)
    return env


def complete_envelope_digest(envelope: dict,
                             cost_binding: dict) -> str:
    """B4-E1: ONE canonical digest over the complete economic
    surface — envelope (headroom included) AND the venue cost
    binding. A geometry-only digest is insufficient."""
    verify_envelope(envelope, cost_binding)
    surface = {"execution_envelope":
               {k: envelope[k] for k in sorted(envelope)},
               "cost_binding":
               {k: cost_binding[k] for k in sorted(cost_binding)}}
    return _sha_obj(surface)


def verify_envelope(envelope: dict, cost_binding: dict) -> None:
    """Refuse BEFORE model or environment construction: missing
    headroom, wrong primitive type, any value other than the one
    rule (the old 0.007102 refuses here)."""
    if not isinstance(envelope, dict):
        raise B4AuthorityRefusal("REFUSED: envelope is not a mapping")
    for k in ENVELOPE_ECONOMIC_KEYS:
        if k not in envelope:
            raise B4AuthorityRefusal(
                f"REFUSED: execution envelope omits economic field "
                f"{k!r}")
    h = envelope["entry_cost_headroom"]
    if type(h) is not float or not math.isfinite(h):
        raise B4AuthorityRefusal(
            "REFUSED: entry_cost_headroom must be a finite float "
            f"(got {type(h).__name__})")
    expected = entry_cost_headroom(cost_binding)
    if h != expected:
        raise B4AuthorityRefusal(
            f"REFUSED: entry_cost_headroom {h} differs from the one "
            f"reviewed rule {expected} (2x per-side + "
            f"{HEADROOM_MARGIN}) — unequal economic envelopes "
            "biased the B0-B3/B4 comparison (B4-E1)")


def verify_language(obj, where: str = "artifact") -> None:
    """B4-E6/E7: new executing artifacts may not carry the
    contradicted authority labels nor name the WP4 smoke path."""
    def _walk(v):
        if isinstance(v, str):
            low = v.lower()
            for phrase in FORBIDDEN_AUTHORITY_PHRASES:
                if phrase in low:
                    raise B4AuthorityRefusal(
                        f"REFUSED: forbidden authority language "
                        f"{phrase!r} in {where}")
        elif isinstance(v, dict):
            for k, x in v.items():
                _walk(k)
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)
    _walk(obj)


def gymfx_lineage_manifest() -> dict:
    """Point-of-use gym-fx lineage: recomputed from the LIVE checkout
    at every execution; a foreign commit (the old 634c3fd3... P1
    runtime included) or a dirty tree refuses."""
    import subprocess
    head = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    if head != GYMFX_PINNED_COMMIT:
        raise B4AuthorityRefusal(
            f"REFUSED: gym-fx checkout {head[:12]} is not the "
            f"accepted lineage {GYMFX_PINNED_COMMIT[:12]} "
            "(satoshi/trade-reconciliation-20260828)")
    dirty = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "status", "--porcelain"],
        capture_output=True, text=True).stdout.strip()
    if dirty:
        raise B4AuthorityRefusal(
            "REFUSED: gym-fx tree is dirty — the point-of-use "
            "manifest must hash the committed lineage only")
    tracked = subprocess.run(
        ["git", "-C", str(GYMFX_REPO), "ls-files", "*.py"],
        capture_output=True, text=True).stdout.split()
    files = {}
    for rel in sorted(tracked):
        fp = GYMFX_REPO / rel
        if fp.exists():
            files[rel] = hashlib.sha256(fp.read_bytes()).hexdigest()
    manifest = {"repo": "gym-fx",
                "branch": "satoshi/trade-reconciliation-20260828",
                "commit": head,
                "files": files}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        manifest, sort_keys=True).encode()).hexdigest()
    return manifest


# --- E4: the complete immutable cell ------------------------------
REQUIRED_CELL_KEYS = (
    # plugin identities
    "env_plugin", "strategy_plugin", "agent_plugin",
    "preprocessor_plugin", "pipeline_plugin",
    # observation + data/split identities
    "feature_columns", "include_price_window", "include_agent_state",
    "agent_state_contract", "window_size", "observation_contract",
    "require_observation_declaration",
    "nested_split_contract_sha256", "source_data_sha256",
    # economics
    "execution_envelope", "complete_envelope_digest",
    "cost_contract_id", "cost_manifest_sha256", "cost_authority",
    "cost_binding", "commission", "slippage_perc",
    # genesis + forbidden inputs
    "genesis_policy", "seed", "train_seed", "eval_seed",
    # SAC parameters consumed
    "net_arch", "learning_rate", "ent_coef", "batch_size",
    "buffer_size", "learning_starts", "train_freq", "gradient_steps",
    "gamma", "tau", "use_sde",
    # epoch/stopping/selection
    "train_days", "epoch_timesteps", "max_epochs", "l1_patience",
    "l1_patience_start_epoch", "l1_min_delta", "selection_metric",
    # action semantics
    "action_space_mode", "continuous_action_threshold",
    "continuous_action_contract", "initial_cash", "solvency_mode",
    # session + lineage + budgets/modes + classification
    "session_exposure_enabled", "gymfx_lineage_manifest_sha256",
    "execution_modes", "output_classification",
)
FORBIDDEN_CELL_KEYS = (
    "warm_start_bundle", "pretrained_branch_generation_dir",
    "checkpoint_bundle_dir", "resume_from", "replay_import")


def verify_cell_complete(cfg: dict) -> None:
    """B4-E4/E7: a cell missing ANY consumed training or budget
    field refuses; hidden pretrained/replay/resume inputs refuse;
    forbidden authority language refuses; the envelope obeys the
    one rule."""
    missing = [k for k in REQUIRED_CELL_KEYS if k not in cfg]
    if missing:
        raise B4AuthorityRefusal(
            f"REFUSED: cell is not a complete runnable experiment — "
            f"missing {missing}")
    hidden = [k for k in FORBIDDEN_CELL_KEYS
              if cfg.get(k) not in (None, "", False)]
    if hidden:
        raise B4AuthorityRefusal(
            f"REFUSED: hidden pretrained/replay/resume inputs {hidden}")
    if cfg.get("session_exposure_enabled") is not False:
        raise B4AuthorityRefusal(
            "REFUSED: session_exposure_enabled must be explicitly "
            "False")
    if cfg.get("genesis_policy", {}).get("warm_start") != "FORBIDDEN":
        raise B4AuthorityRefusal(
            "REFUSED: genesis policy must declare warm_start "
            "FORBIDDEN")
    binding = cfg["cost_binding"]
    if (binding.get("commission") != cfg.get("commission")
            or binding.get("slippage_perc")
            != cfg.get("slippage_perc")):
        raise B4AuthorityRefusal(
            "REFUSED: cell cost_binding disagrees with its own "
            "flattened cost fields")
    verify_envelope(cfg["execution_envelope"], binding)
    declared = complete_envelope_digest(cfg["execution_envelope"],
                                        binding)
    if cfg["complete_envelope_digest"] != declared:
        raise B4AuthorityRefusal(
            "REFUSED: complete_envelope_digest does not re-derive "
            "from the cell's own envelope and costs")
    modes = cfg["execution_modes"]
    if "cpu_mechanics_replay" not in modes:
        raise B4AuthorityRefusal(
            "REFUSED: cell without the bounded cpu_mechanics_replay "
            "mode")
    for req in ("budget_max_env_steps", "budget_max_updates",
                "budget_max_wall_seconds", "rss_cap_bytes",
                "thermal_cap_celsius", "stop_file_policy",
                "train_role", "train_year", "replay_buffer_cap",
                "learn_segments"):
        if req not in modes["cpu_mechanics_replay"]:
            raise B4AuthorityRefusal(
                f"REFUSED: mechanics mode omits budget field {req!r}")
    verify_language(cfg, "cell effective_config")


# --- E3: the executable amendment chain ---------------------------
def verify_amendment_chain() -> dict:
    """Establish the ordered chain design -> amendments 1-3 ->
    amendment 4 and return the FINAL executing-code pins. A missing,
    reordered or altered chain refuses; amendment 4 must name the
    design and the exact prior chain, and its pins must equal the
    live files."""
    if not DESIGN_PATH.is_file():
        raise B4AuthorityRefusal("REFUSED: sealed design absent")
    if _sha_file(DESIGN_PATH) != DESIGN_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: sealed design bytes differ from the reviewed "
            "identity — historical artifacts are immutable")
    for i, (p, want) in enumerate(zip(AMENDMENT_PATHS,
                                      AMENDMENT_SHAS), start=1):
        if not p.is_file():
            raise B4AuthorityRefusal(
                f"REFUSED: amendment {i} absent — the chain is "
                "append-only and complete")
        got = _sha_file(p)
        if got != want:
            raise B4AuthorityRefusal(
                f"REFUSED: amendment {i} bytes {got[:12]} differ "
                f"from the carried chain identity {want[:12]}")
    if not AMENDMENT_4_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: final amendment (4) absent — no executable "
            "repair of the design/code boundary exists")
    a4 = json.loads(AMENDMENT_4_PATH.read_text())
    if a4.get("amends_design_sha256") != DESIGN_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 4 does not name the sealed design")
    if tuple(a4.get("supersedes_amendment_shas", ())) != \
            AMENDMENT_SHAS:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 4 names a different or reordered "
            "prior chain")
    pins = a4.get("final_code_pins", {})
    for rel, want in pins.items():
        live = _sha_file(REPO / rel)
        if live != want:
            raise B4AuthorityRefusal(
                f"REFUSED: executing code {rel} digest {live[:12]} "
                f"differs from the final amendment pin {want[:12]}")
    required_pins = {"tools/b4_authority.py",
                     "tools/screen_b_baselines.py",
                     "tools/materialize_b4_causal_sac.py",
                     "tools/b4_run_cell.py"}
    if not required_pins.issubset(pins):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 4 does not pin the full executing "
            "surface")
    return {"design_sha256": DESIGN_SHA,
            "amendment_shas": list(AMENDMENT_SHAS)
            + [_sha_file(AMENDMENT_4_PATH)],
            "final_code_pins": pins,
            "design": json.loads(DESIGN_PATH.read_bytes())}


# --- E5: evidence-complete comparator verification ----------------
EXPECTED_ARMS = ("B0", "B1", "B2a", "B2b", "B3")
EXPECTED_ORIGINS = (2022, 2023, 2024)
EXPECTED_COST_SET = "alpaca_ethusd"
EXPECTED_CAL_GEOMS = 7
EXPECTED_CAL_ARMS = 4


def verify_comparator_population(baselines_dir: Path,
                                 design: dict) -> dict:
    """Consume the run manifest, ledger, 15 result records, frozen
    envelope artifacts and referenced digests; RE-DERIVE cardinality,
    coverage, terminal state and the selected envelope per origin.
    Labels and supplied counts grant nothing."""
    d = Path(baselines_dir)
    manifest_p = d / "RUN_MANIFEST.json"
    results_p = d / "SCREEN_B_RESULTS.json"
    ledger_p = d / "trial_ledger.jsonl"
    for p in (manifest_p, results_p, ledger_p):
        if not p.is_file():
            raise B4AuthorityRefusal(
                f"REFUSED: comparator evidence incomplete — {p.name} "
                "absent")
    manifest = json.loads(manifest_p.read_bytes())
    packet = json.loads(results_p.read_bytes())
    if packet.get("run_manifest_sha256") != _sha_file(manifest_p):
        raise B4AuthorityRefusal(
            "REFUSED: result packet does not bind this run manifest")
    if manifest.get("superseding_design_sha256") != DESIGN_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: comparator was not scored under the sealed "
            "superseding design")
    lineage = gymfx_lineage_manifest()
    if manifest.get("gymfx_lineage_manifest_sha256") != \
            lineage["manifest_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: comparator lineage differs from the live "
            "point-of-use gym-fx manifest")
    if manifest.get("source_data_sha256") != \
            design["source_data_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: comparator data differs from the sealed design")
    # frozen per-origin envelope artifacts re-derived
    frozen = {}
    for year in EXPECTED_ORIGINS:
        calf = d / f"ENVELOPE_CALIBRATION_o{year}.json"
        if not calf.is_file():
            raise B4AuthorityRefusal(
                f"REFUSED: frozen envelope artifact absent for "
                f"origin {year}")
        cal = json.loads(calf.read_bytes())
        geom = cal["frozen_geometry"]
        if _sha_obj(geom) != cal["frozen_envelope_sha256"]:
            raise B4AuthorityRefusal(
                f"REFUSED: origin {year} frozen envelope digest does "
                "not re-derive from its geometry")
        cells = cal["grid_cells"]
        if len(cells) != EXPECTED_CAL_GEOMS:
            raise B4AuthorityRefusal(
                f"REFUSED: origin {year} calibration grid has "
                f"{len(cells)} cells, expected {EXPECTED_CAL_GEOMS}")
        winner = [c for c in cells if c["envelope_sha256"]
                  == cal["frozen_envelope_sha256"]]
        if len(winner) != 1 or not \
                winner[0]["criterion"].get("eligible"):
            raise B4AuthorityRefusal(
                f"REFUSED: origin {year} frozen geometry is not the "
                "eligible winner of its own grid")
        if cal.get("calibration_year") != year - 1:
            raise B4AuthorityRefusal(
                f"REFUSED: origin {year} calibrated on "
                f"{cal.get('calibration_year')}, not the causal "
                "year-1 window")
        frozen[year] = cal
    # ledger re-derivation
    rows = [json.loads(line) for line in
            ledger_p.read_text().splitlines() if line.strip()]
    ids = [r["trial_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise B4AuthorityRefusal("REFUSED: duplicate ledger trials")
    if any(not r.get("registered_before_results") for r in rows):
        raise B4AuthorityRefusal(
            "REFUSED: a ledger trial was not registered before "
            "results")
    cal_rows = [r for r in rows
                if r.get("screen") == "B_envelope_calibration"]
    score_rows = [r for r in rows if r.get("screen") == "B"]
    if len(cal_rows) != (EXPECTED_CAL_GEOMS * EXPECTED_CAL_ARMS
                         * len(EXPECTED_ORIGINS)):
        raise B4AuthorityRefusal(
            f"REFUSED: calibration ledger cardinality "
            f"{len(cal_rows)} != expected "
            f"{EXPECTED_CAL_GEOMS * EXPECTED_CAL_ARMS * 3}")
    if len(score_rows) != len(EXPECTED_ARMS) * len(EXPECTED_ORIGINS):
        raise B4AuthorityRefusal(
            f"REFUSED: score ledger cardinality {len(score_rows)} "
            "!= 15")
    if len(rows) != len(cal_rows) + len(score_rows):
        raise B4AuthorityRefusal(
            "REFUSED: ledger carries unclassified trials")
    # the 15 results re-derived, one per (arm, origin), digests live
    results = packet.get("results", [])
    if len(results) != 15:
        raise B4AuthorityRefusal(
            f"REFUSED: comparator population has {len(results)} "
            "results, expected exactly 15")
    seen = set()
    ledger_score_ids = {r["trial_id"] for r in score_rows}
    for r in results:
        key = (r.get("arm"), int(r.get("origin", 0)),
               r.get("cost_set"))
        if key in seen:
            raise B4AuthorityRefusal(
                f"REFUSED: duplicate result cell {key}")
        seen.add(key)
        if r.get("cost_set") != EXPECTED_COST_SET:
            raise B4AuthorityRefusal(
                f"REFUSED: non-G1 cost set in population: {key}")
        if r.get("population_label") != \
                packet.get("population_label"):
            raise B4AuthorityRefusal(
                "REFUSED: result population label mismatch")
        if r.get("gymfx_lineage_manifest_sha256") != \
                lineage["manifest_sha256"]:
            raise B4AuthorityRefusal(
                f"REFUSED: result {key} carries a foreign lineage")
        year = int(r["origin"])
        if r.get("execution_envelope_sha256") != \
                frozen[year]["frozen_envelope_sha256"]:
            raise B4AuthorityRefusal(
                f"REFUSED: result {key} did not run under origin "
                f"{year}'s frozen envelope")
        if r.get("trial_id") not in ledger_score_ids:
            raise B4AuthorityRefusal(
                f"REFUSED: result {key} has no pre-registered "
                "ledger trial")
        pb = Path(r.get("per_bar_csv", ""))
        if not pb.is_file() or _sha_file(pb) != \
                r.get("per_bar_sha256"):
            raise B4AuthorityRefusal(
                f"REFUSED: result {key} per-bar evidence missing or "
                "digest-broken")
        if "complete_envelope_digest" not in r:
            raise B4AuthorityRefusal(
                f"REFUSED: result {key} lacks the complete-envelope "
                "digest (B4-E1)")
        verify_language(
            {"cost_authority": r.get("cost_authority", "")},
            f"comparator result {key}")
    expected_cells = {(a, y, EXPECTED_COST_SET)
                      for a in EXPECTED_ARMS
                      for y in EXPECTED_ORIGINS}
    if seen != expected_cells:
        raise B4AuthorityRefusal(
            f"REFUSED: population coverage mismatch — missing "
            f"{sorted(expected_cells - seen)}")
    if packet.get("sealed_2025_used") is not False:
        raise B4AuthorityRefusal(
            "REFUSED: packet does not prove sealed-2025 absence")
    return {"lineage": lineage, "frozen_by_origin": frozen,
            "population_label": packet.get("population_label"),
            "n_results": len(results), "n_ledger": len(rows)}


def verify_full_authority_chain(baselines_dir: Path) -> dict:
    """B4-E3 §5: the ONE verifier — establishes, in order: the
    design + amendment chain and final code pins (1, 2), the live
    gym-fx point-of-use manifest (3), the owner-ratified observation
    v2 identity (4), the fixed Alpaca G1 cost identity (5), the
    complete-envelope rule (6, enforced per artifact by
    verify_envelope), data + causal split identities (7) and the
    exact comparator population (8)."""
    chain = verify_amendment_chain()
    design = chain["design"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "n4a_auth", REPO / "tools/n4_target_audit.py")
    n4a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n4a)
    n4a.verify_owner_act()
    cost_p = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
              "cost_manifest_eth_h4_v2_screen_b_20260826.json")
    if _sha_file(cost_p) != design["cost_manifest_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: fixed experimental cost model bytes differ "
            "from the sealed design")
    data_p = Path(design.get(
        "source_data_path",
        "/home/harveybc/Documents/GitHub/predictor/examples/data/"
        "project3/ethusdt_4h_tech_stat_full_model_ready.csv"))
    if _sha_file(data_p) != design["source_data_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: source dataset bytes differ from the sealed "
            "design")
    comparator = verify_comparator_population(baselines_dir, design)
    return {"chain": chain, "comparator": comparator,
            "design": design}
