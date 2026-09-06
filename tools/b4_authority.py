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
AMENDMENT_5_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_5_2026_09_05"
                    ".json")
AMENDMENT_6_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_6_2026_09_05"
                    ".json")
AMENDMENT_7_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_7_2026_09_06"
                    ".json")
AMENDMENT_8_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_8_2026_09_06"
                    ".json")
AMENDMENT_9_PATH = (EVIDENCE /
                    "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_9_2026_09_06"
                    ".json")
# C26: amendment 9's bytes are HISTORY — restored to the reviewed
# blob at d97c3f62 and pinned by digest; any in-place rewrite
# refuses. Later corrections append (amendment 10+), never edit.
AMENDMENT_9_SHA = ("eb9d49707b2a173056b07c3802b617d8302b38e5f422"
                   "45efed7db5e8d155ca42")
AMENDMENT_10_PATH = (EVIDENCE /
                     "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_10_"
                     "2026_09_06.json")
# C27: amendment 10 is byte-immutable HISTORY like amendment 9 —
# the reviewer authorization record binds this exact snapshot.
AMENDMENT_10_SHA = ("c299d03ee1bab35c14094023ca88486e0cdee6e1b5"
                    "313d8a405f6e3d569e3848")
AMENDMENT_11_PATH = (EVIDENCE /
                     "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_11_"
                     "2026_09_06.json")
# C27: the two reviewer-authored records, byte-pinned. The
# candidate never authors either.
OWNER_RATIFICATION_PATH = (
    EVIDENCE /
    "OWNER_B4_CAMPAIGN_AUTHORIZATION_RATIFIED_2026_09_06.json")
OWNER_RATIFICATION_SHA = (
    "540fb175f0203338aa08a21bc91bba1dddf942008e011ab7b34c6695"
    "af776c63")
CAMPAIGN_AUTHORIZATION_RECORD_PATH = (
    EVIDENCE / "MUSASHI_B4_CAMPAIGN_AUTHORIZATION_RECORD.json")

# --- B4-P1 (order @9fb017e3): the exact owner GPU authorization ---
# Fixed reviewed identities: neither the path nor the digest can come
# from CLI, environment, materialization root or output root.
OWNER_GPU_AUTH_PATH = (
    EVIDENCE /
    "OWNER_AUTHORIZATION_B4_SINGLE_GPU_PREFLIGHT_2026_09_05.json")
OWNER_GPU_AUTH_SHA = ("7426a0bfc9cdb6c609730755512c0936f58fdc45d"
                      "40cd17c0bb5a83ce00cdf82")
GPU_ATTEMPT_LEDGER = (Path.home() / ".local/share/agent-multi/"
                      "b4_gpu_preflight_attempt_ledger.json")

# --- C3 (order @0ce52740): the AUTHORIZED resource contract is a
# carried identity; effective limits derive from IT, never from the
# cell's old gpu_economic mode. The future Musashi authorization
# record must bind these exact digests.
RESOURCE_CONTRACT_PATH = (
    EVIDENCE / "B4_CAMPAIGN_RESOURCE_CONTRACT_PROPOSAL_2026_09_05"
               ".json")
RESOURCE_CONTRACT_SHA = ("1b738f74534fa4ca6fc88e5373caec9aaac7be44"
                         "96a60547e19fb66c3b13cdaf")
# C9: the SUPERSEDING v2 contract (the v1 proposal named amendment 6
# and a retired population). Stamped by the sealing script; the sha
# is injected at authoring like every carried identity.
RESOURCE_CONTRACT_V2_PATH = (
    EVIDENCE / "B4_CAMPAIGN_RESOURCE_CONTRACT_V2_2026_09_06.json")
RESOURCE_CONTRACT_V2_SHA = "516bd7d70e46353c2e4fec00761d4322511127c222d9432928cc8f723bec0b20"
# C9 (order 2026-09-06): the record bindings DERIVE from the LIVE
# amendment chain at verification time — never fossilized constants.
# The latest amendment's digest is bound under its TRUTHFUL field
# name (amendment_7_sha256 today; a future amendment renames it).
CAMPAIGN_GENERATION = "b4_campaign_generation_v5_20260906"


def campaign_record_required_bindings() -> dict:
    """C27 (non-circular by design): the reviewer authorization
    record binds the audited PRE-ACTIVATION amendment-10 snapshot
    (immutable constant), while the live chain advances to
    amendment 11 which binds that reviewer record and the final
    consuming code. The record never pre-binds amendment 11."""
    chain = verify_amendment_chain()
    pop = chain["proposed_campaign_population"]
    if not pop:
        raise B4AuthorityRefusal(
            "REFUSED: the live chain carries no proposed campaign "
            "population")
    return {
        "cell_population_sha256": pop["cell_population_sha256"],
        "materialization_sha256": pop["materialization_sha256"],
        "genesis_binding_sha256": pop["genesis_binding_sha256"],
        "amendment_10_sha256": AMENDMENT_10_SHA,
        "resource_contract_sha256": RESOURCE_CONTRACT_V2_SHA,
        "campaign_generation": CAMPAIGN_GENERATION,
    }


def _strict_json_bytes(raw: bytes, where: str) -> dict:
    """C9: duplicate-key and non-finite rejection at parse."""
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise B4AuthorityRefusal(
                f"REFUSED: duplicate JSON key in {where}")
        return dict(pairs)
    try:
        return json.loads(raw.decode("utf-8"),
                          object_pairs_hook=_no_dupes,
                          parse_constant=lambda c: (_ for _ in ()
                                                    ).throw(
                              B4AuthorityRefusal(
                                  f"REFUSED: non-finite literal "
                                  f"{c!r} in {where}")))
    except json.JSONDecodeError as exc:
        raise B4AuthorityRefusal(
            f"REFUSED: invalid JSON in {where}: {exc}")


def _canonical_sha(v, field: str) -> str:
    if type(v) is not str or len(v) != 64 or \
            v != v.lower() or any(c not in "0123456789abcdef"
                                  for c in v):
        raise B4AuthorityRefusal(
            f"REFUSED: {field} is not a canonical lowercase 64-hex "
            "SHA-256 (no normalization is applied)")
    return v


def load_resource_contract() -> dict:
    """Hash-before-parse consumption of the AUTHORIZED resource
    contract; the effective per-cell limits every executor path must
    install. Consumes the SUPERSEDING v2 when stamped, else v1."""
    if not RESOURCE_CONTRACT_V2_SHA.startswith("@"):
        raw = RESOURCE_CONTRACT_V2_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != \
                RESOURCE_CONTRACT_V2_SHA:
            raise B4AuthorityRefusal(
                "REFUSED: resource contract v2 bytes differ from "
                "the carried digest")
        c = _strict_json_bytes(raw, "resource contract v2")
        lim = c["per_cell_limits"]
        return {
            "budget_max_env_steps":
                int(lim["environment_steps_max"]),
            "budget_max_updates": int(lim["real_updates_max"]),
            "budget_max_wall_seconds":
                float(lim["wall_seconds_max"]),
            "budget_max_rss_bytes": int(lim["host_rss_bytes_max"]),
            "budget_max_cuda_bytes": int(
                lim["cuda_allocated_bytes_max"]),
            "budget_max_gpu_temp_celsius": float(
                lim["gpu_temperature_celsius_max"]),
            "global_gpu_hours_ceiling": float(
                c["global_limits"]["gpu_hours_ceiling_all_cells"]),
            "max_concurrency": int(
                c["global_limits"]["max_concurrency"]),
            "heartbeat_seconds": int(
                c["monitoring"]["heartbeat_seconds"]),
            "gpu_telemetry_sample_seconds": float(
                c["monitoring"]["gpu_telemetry_sample_seconds"]),
        }
    raw = RESOURCE_CONTRACT_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != RESOURCE_CONTRACT_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: resource contract bytes differ from the "
            "carried authorized digest")
    c = json.loads(raw)
    lim = c["per_cell_limits"]
    return {
        "budget_max_env_steps": int(lim["environment_steps_max"]),
        "budget_max_updates": int(lim["real_updates_max"]),
        "budget_max_wall_seconds": float(lim["wall_seconds_max"]),
        "budget_max_rss_bytes": int(lim["host_rss_bytes_max"]),
        "budget_max_cuda_bytes": int(
            lim["cuda_allocated_bytes_max"]),
        "budget_max_gpu_temp_celsius": float(
            lim["gpu_temperature_celsius_max"]),
        "global_gpu_hours_ceiling": float(
            c["global_limits"]["gpu_hours_ceiling_all_cells"]),
        "max_concurrency": int(c["global_limits"]["max_concurrency"]),
        "heartbeat_seconds": int(
            c["monitoring"]["heartbeat_seconds"]),
    }


def verify_campaign_authorization_record(path: Path,
                                         expected_sha: str) -> dict:
    """C9: the future Musashi record — STRICT schema (exact keys,
    exact primitive types, duplicate-key and non-finite rejection,
    canonical lowercase hex), full-digest equality against the LIVE
    chain bindings, exact authorized limits and identities. Any
    deviation refuses BEFORE model, env, CUDA or output."""
    path = Path(path)
    if not path.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: campaign authorization record absent")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise B4AuthorityRefusal(
            "REFUSED: campaign authorization bytes differ from the "
            "carried reviewed digest")
    rec = _strict_json_bytes(raw, "campaign authorization record")
    required_top = {"schema", "recorded_at_date", "authority",
                    "recorded_by", "decision", "bindings",
                    "per_cell_limits", "owner_decision"}
    if set(rec) != required_top:
        raise B4AuthorityRefusal(
            f"REFUSED: record keys must be exactly "
            f"{sorted(required_top)}")
    for k in ("schema", "recorded_at_date", "authority",
              "recorded_by", "decision"):
        if type(rec[k]) is not str or not rec[k]:
            raise B4AuthorityRefusal(
                f"REFUSED: record field {k!r} must be a nonempty "
                "string")
    if rec["decision"] != "APPROVE_B4_TWELVE_CELL_CAMPAIGN":
        raise B4AuthorityRefusal(
            "REFUSED: the decision is not the twelve-cell campaign "
            "approval")
    od = rec["owner_decision"]
    if not isinstance(od, dict) or set(od) != {
            "intent_record_sha256", "owner_words"}:
        raise B4AuthorityRefusal(
            "REFUSED: owner_decision must carry exactly the intent "
            "record digest and the owner's words")
    _canonical_sha(od["intent_record_sha256"],
                   "owner intent digest")
    # C27.6: the ratification record the digest names must EXIST,
    # hash to that digest, carry the exact owner words, and scope
    # this twelve-cell campaign. A canonical-looking but absent or
    # unrelated intent digest refuses.
    rat_p = OWNER_RATIFICATION_PATH
    if not rat_p.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: the owner ratification record named by the "
            "authorization is absent")
    rat_raw = rat_p.read_bytes()
    if hashlib.sha256(rat_raw).hexdigest() != \
            od["intent_record_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: the owner ratification bytes do not hash to "
            "the intent digest the authorization names — an "
            "unrelated object grants nothing")
    rat = _strict_json_bytes(rat_raw, "owner ratification record")
    if rat.get("owner_words") != od["owner_words"]:
        raise B4AuthorityRefusal(
            "REFUSED: the owner words differ between the "
            "authorization and the ratification record")
    scope = rat.get("scope")
    if not isinstance(scope, list) or not any(
            isinstance(s, str) and "twelve-cell B4 population" in s
            for s in scope):
        raise B4AuthorityRefusal(
            "REFUSED: the ratification scope does not cover this "
            "twelve-cell campaign")
    if not isinstance(rat.get("does_not_authorize"), list) or \
            not rat["does_not_authorize"]:
        raise B4AuthorityRefusal(
            "REFUSED: the ratification lacks its explicit "
            "non-authorization boundary")
    binds = rec["bindings"]
    expected_binds = campaign_record_required_bindings()
    if not isinstance(binds, dict) or \
            set(binds) != set(expected_binds):
        raise B4AuthorityRefusal(
            f"REFUSED: bindings keys must be exactly "
            f"{sorted(expected_binds)} (truthful latest-amendment "
            "field name included)")
    for k, want in expected_binds.items():
        got = binds[k]
        if k != "campaign_generation":
            _canonical_sha(got, f"binding {k}")
        if got != want:
            raise B4AuthorityRefusal(
                f"REFUSED: campaign record binding {k!r} differs "
                "from the LIVE chain identity")
    limits = load_resource_contract()
    rec_lim = rec["per_cell_limits"]
    lim_keys = ("budget_max_env_steps", "budget_max_updates",
                "budget_max_wall_seconds", "budget_max_rss_bytes",
                "budget_max_cuda_bytes",
                "budget_max_gpu_temp_celsius")
    if not isinstance(rec_lim, dict) or \
            set(rec_lim) != set(lim_keys):
        raise B4AuthorityRefusal(
            "REFUSED: per_cell_limits keys must be exactly the six "
            "authorized limits")
    for k in lim_keys:
        want = limits[k]
        got = rec_lim[k]
        if type(got) is not type(want) or got != want:
            raise B4AuthorityRefusal(
                f"REFUSED: record limit {k!r} differs in value or "
                "primitive type from the authorized contract")
    return {"record": rec, "limits": limits,
            "bindings": expected_binds}


def author_campaign_record_template(out: Path) -> Path:
    """C9: the machine-checkable template with the EXACT digests
    Musashi must review — the candidate never authors, installs or
    pins the final record."""
    binds = campaign_record_required_bindings()
    limits = load_resource_contract()
    template = {
        "schema": "agent_multi.owner_campaign_authorization.v2",
        "recorded_at_date": "<MUSASHI_FILLS_DATE>",
        "authority": "project_owner",
        "recorded_by": "General Musashi",
        "decision": "APPROVE_B4_TWELVE_CELL_CAMPAIGN",
        "bindings": binds,
        "per_cell_limits": {k: limits[k] for k in (
            "budget_max_env_steps", "budget_max_updates",
            "budget_max_wall_seconds", "budget_max_rss_bytes",
            "budget_max_cuda_bytes",
            "budget_max_gpu_temp_celsius")},
        "owner_decision": {
            "intent_record_sha256": "<MUSASHI_VERIFIES_INTENT_SHA>",
            "owner_words": "ok yo autorizo"},
    }
    out.write_text(json.dumps(template, indent=1))
    return out


# --- F7 (order @0ce52740): public evidence carries LOGICAL
# identities; local roots resolve at point of use only.
def resolve_predictor_root() -> Path:
    import os
    root = os.environ.get("B4_PREDICTOR_ROOT")
    if root:
        p = Path(root)
    else:
        p = REPO.parent.parent / "predictor"
        if not p.is_dir():
            p = Path.home() / "Documents/GitHub/predictor"
    if not p.is_dir():
        raise B4AuthorityRefusal(
            "REFUSED: predictor data root unresolved — set "
            "B4_PREDICTOR_ROOT")
    return p


def resolve_state_root() -> Path:
    import os
    root = os.environ.get("B4_STATE_ROOT")
    p = (Path(root) if root
         else Path.home() / ".local/share/agent-multi")
    if not p.is_dir():
        raise B4AuthorityRefusal(
            "REFUSED: state root unresolved — set B4_STATE_ROOT")
    return p


def resolve_source_ref(ref: str) -> Path:
    """A logical source_ref like 'predictor:examples/data/...' or
    'repo:docs/...' resolves under an explicit root at runtime."""
    if ref.startswith("predictor:"):
        return resolve_predictor_root() / ref.split(":", 1)[1]
    if ref.startswith("repo:"):
        return REPO / ref.split(":", 1)[1]
    if ref.startswith("state:"):
        return resolve_state_root() / ref.split(":", 1)[1]
    raise B4AuthorityRefusal(
        f"REFUSED: unknown logical source ref {ref!r}")


ABS_PATH_MARKERS = ("/home/", "/Users/", "C:\\")


def verify_no_absolute_paths(obj, where: str = "artifact") -> None:
    """F7: committed public evidence may not carry absolute local
    paths, usernames, hosts or storage topology."""
    def _walk(v):
        if isinstance(v, str):
            for m in ABS_PATH_MARKERS:
                if m in v:
                    raise B4AuthorityRefusal(
                        f"REFUSED: absolute local path in {where}: "
                        f"{v[:60]!r}")
        elif isinstance(v, dict):
            for k, x in v.items():
                _walk(k)
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)
    _walk(obj)

_AUTH_SCHEMA = {
    "schema": str, "recorded_at_date": str, "authority": str,
    "recorded_by": str, "resolves": dict, "decision": str,
    "approved_cell": dict, "approved_comparator": dict,
    "approved_scientific_identity": dict, "preflight_limits": dict,
    "execution_contract": dict, "preconditions": list,
    "does_not_authorize": list}


def verify_gpu_preflight_authorization() -> dict:
    """Hash-before-parse consumption of the exact owner record, then
    exact schema, exact primitive types and exact agreement with the
    carried scientific identities. Any deviation refuses BEFORE CUDA
    initialization, model construction or output creation."""
    if not OWNER_GPU_AUTH_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: owner GPU authorization absent — no GPU path "
            "exists without the exact owner record")
    raw = OWNER_GPU_AUTH_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OWNER_GPU_AUTH_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: owner GPU authorization bytes differ from the "
            "carried reviewed digest — an edited, self-rehashed or "
            "substituted record grants nothing")
    rec = json.loads(raw)
    for k, t in _AUTH_SCHEMA.items():
        if k not in rec or type(rec[k]) is not t:
            raise B4AuthorityRefusal(
                f"REFUSED: owner record field {k!r} missing or "
                "mistyped")
    if set(rec) != set(_AUTH_SCHEMA):
        raise B4AuthorityRefusal(
            "REFUSED: owner record carries unknown top-level fields")
    if rec["decision"] != "APPROVE_ONE_B4_BOUNDED_GPU_PREFLIGHT_ONLY":
        raise B4AuthorityRefusal(
            "REFUSED: the recorded decision is not the single "
            "bounded GPU preflight approval")
    ident = rec["approved_scientific_identity"]
    if ident["superseding_design_sha256"] != DESIGN_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: owner record names a different sealed design")
    if ident["gymfx_commit"] != GYMFX_PINNED_COMMIT:
        raise B4AuthorityRefusal(
            "REFUSED: owner record names a different gym-fx lineage")
    if ident["cost_authority"] !=             "MUSASHI_REVIEWED_FIXED_EXPERIMENTAL_COST_MODEL":
        raise B4AuthorityRefusal(
            "REFUSED: owner record names a different cost authority")
    if type(ident["complete_entry_cost_headroom"]) is not float or             ident["complete_entry_cost_headroom"] != 0.012102:
        raise B4AuthorityRefusal(
            "REFUSED: owner record headroom differs from the one "
            "reviewed rule")
    lim = rec["preflight_limits"]
    for k, t in (("attempts", int), ("environment_steps_max", int),
                 ("optimizer_updates_max", int),
                 ("wall_seconds_max", int),
                 ("host_rss_bytes_max", int),
                 ("cuda_allocated_bytes_max", int),
                 ("gpu_temperature_celsius_max", int),
                 ("stop_file_required", bool),
                 ("heartbeat_seconds_max", int)):
        if type(lim.get(k)) is not t:
            raise B4AuthorityRefusal(
                f"REFUSED: preflight limit {k!r} missing or mistyped")
    if lim["attempts"] != 1:
        raise B4AuthorityRefusal(
            "REFUSED: the owner approved exactly ONE attempt")
    if list(lim["learning_segments"]) != [
            lim["environment_steps_max"]]:
        raise B4AuthorityRefusal(
            "REFUSED: learning segments differ from the single "
            "bounded segment the owner approved")
    return rec


def verify_approved_materialization(mat_root: Path,
                                    rec: dict) -> None:
    """B4-P2: hash and compare the exact files the owner record
    names BEFORE parsing any cell — a self-consistent replacement
    tree (repaired internal digests included) grants nothing."""
    mat_root = Path(mat_root)
    cell = rec["approved_cell"]
    pins = (
        ("B4_CELL_CONFIGS.json", cell["cell_population_sha256"]),
        ("B4_MATERIALIZATION.json", cell["materialization_sha256"]),
        ("genesis/GENESIS_BINDING.json",
         cell["genesis_binding_sha256"]),
    )
    for rel, want in pins:
        f = mat_root / rel
        if not f.is_file():
            raise B4AuthorityRefusal(
                f"REFUSED: approved artifact {rel} absent from the "
                "materialization root")
        got = _sha_file(f)
        if got != want:
            raise B4AuthorityRefusal(
                f"REFUSED: {rel} digest {got[:12]} differs from the "
                f"owner-approved identity {want[:12]} — the "
                "externally pinned population is not this tree")
    cells = json.loads((mat_root / "B4_CELL_CONFIGS.json").read_bytes())
    entry = cells.get(cell["cell_id"])
    if not entry:
        raise B4AuthorityRefusal(
            "REFUSED: approved cell absent from the population")
    canonical = hashlib.sha256(json.dumps(
        entry["effective_config"], sort_keys=True,
        default=str).encode()).hexdigest()
    if canonical != cell["cell_config_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: the selected cell's canonical config differs "
            "from the owner-approved cell identity")


def verify_approved_comparator(baselines_dir: Path,
                               rec: dict) -> None:
    """B4-P1: the comparator the owner approved, by exact digests."""
    d = Path(baselines_dir)
    comp = rec["approved_comparator"]
    for rel, want in (("RUN_MANIFEST.json",
                       comp["run_manifest_sha256"]),
                      ("SCREEN_B_RESULTS.json",
                       comp["results_sha256"]),
                      ("trial_ledger.jsonl",
                       comp["trial_ledger_sha256"])):
        got = _sha_file(d / rel)
        if got != want:
            raise B4AuthorityRefusal(
                f"REFUSED: comparator {rel} digest differs from the "
                "owner-approved population")

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
# checkpoint_bundle_dir is a pipeline OUTPUT (the coherent
# per-improvement bundle, findings 307/308/309) derived per-cell by
# the executor — the actual resume inputs are the keys below.
FORBIDDEN_CELL_KEYS = (
    "warm_start_bundle", "pretrained_branch_generation_dir",
    "resume_from", "resume_from_cell_runtime", "replay_import")


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
    pins = dict(a4.get("final_code_pins", {}))
    required_pins = {"tools/b4_authority.py",
                     "tools/screen_b_baselines.py",
                     "tools/materialize_b4_causal_sac.py",
                     "tools/b4_run_cell.py"}
    if not required_pins.issubset(pins):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 4 does not pin the full executing "
            "surface")
    # B4-P1 (order @9fb017e3): amendment 5 is the execution-only
    # append-only step — it names amendment 4 and the owner-record
    # digest, changes no scientific parameter or data role, and its
    # pins SUPERSEDE amendment 4's for the files it re-pins.
    if not AMENDMENT_5_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 5 absent — the execution chain "
            "through the owner GPU authorization does not exist")
    a5 = json.loads(AMENDMENT_5_PATH.read_bytes())
    if a5.get("amends_amendment_4_sha256") != _sha_file(
            AMENDMENT_4_PATH):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 5 does not name amendment 4's exact "
            "bytes")
    if a5.get("owner_authorization_sha256") != OWNER_GPU_AUTH_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 5 does not name the owner-record "
            "digest")
    if a5.get("scientific_change") != "NONE":
        raise B4AuthorityRefusal(
            "REFUSED: amendment 5 must declare zero scientific or "
            "data-role change")
    a5_pins = a5.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a5_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 5 does not pin the final "
                "authority module, runner and tests")
    pins.update(a5_pins)
    # E8-E12 (order @e8bb500f): amendment 6 is the campaign-
    # preparation append-only step — it names amendment 5's exact
    # bytes, DISCLOSES its data-role change (the per-origin sealed
    # zone correction; never silent), carries the PROPOSED campaign
    # population identities for the owner's future authorization,
    # and its pins supersede amendment 5's for the files it re-pins.
    campaign_pins = None
    if not AMENDMENT_6_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 6 absent — the campaign-preparation "
            "chain does not exist")
    a6 = json.loads(AMENDMENT_6_PATH.read_bytes())
    if a6.get("amends_amendment_5_sha256") != _sha_file(
            AMENDMENT_5_PATH):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 6 does not name amendment 5's "
            "exact bytes")
    if not a6.get("data_role_change_disclosure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 6 must disclose its data-role "
            "change explicitly")
    campaign_pins = a6.get("proposed_campaign_population")
    a6_pins = a6.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a6_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 6 does not pin the full "
                "campaign executing surface")
    pins.update(a6_pins)
    # C8 (order @0ce52740): amendment 7 is the runtime-correction
    # append-only step — names amendment 6's exact bytes, discloses
    # its artifact-identity change, carries the CORRECTED campaign
    # population and supersedes pins for the runtime surface.
    if not AMENDMENT_7_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 7 absent — the corrected runtime "
            "chain does not exist")
    a7 = json.loads(AMENDMENT_7_PATH.read_bytes())
    if a7.get("amends_amendment_6_sha256") != _sha_file(
            AMENDMENT_6_PATH):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 7 does not name amendment 6's "
            "exact bytes")
    if not a7.get("change_disclosure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 7 must disclose its changes")
    if a7.get("proposed_campaign_population"):
        campaign_pins = a7["proposed_campaign_population"]
    a7_pins = a7.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_campaign_orchestrator.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "pipeline_plugins/rl_pipeline_with_validation.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a7_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 7 does not pin the corrected "
                "runtime surface")
    pins.update(a7_pins)
    # C9 (audit 2026-09-06): amendment 8 — the runtime-authority
    # correction generation; names a7's exact bytes, discloses, and
    # carries the CURRENT population; its pins supersede.
    if not AMENDMENT_8_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 8 absent — the corrected authority "
            "generation does not exist")
    a8 = json.loads(AMENDMENT_8_PATH.read_bytes())
    if a8.get("amends_amendment_7_sha256") != _sha_file(
            AMENDMENT_7_PATH):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 8 does not name amendment 7's "
            "exact bytes")
    if not a8.get("change_disclosure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 8 must disclose its changes")
    if a8.get("proposed_campaign_population"):
        campaign_pins = a8["proposed_campaign_population"]
    a8_pins = a8.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_campaign_orchestrator.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "pipeline_plugins/rl_pipeline_with_validation.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a8_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 8 does not pin the corrected "
                "runtime surface")
    pins.update(a8_pins)
    # C17-C22 (final audit 2026-09-06): amendment 9 — the capability/
    # seal/factual-verifier generation; names a8's exact bytes,
    # discloses, carries the population; its pins supersede.
    if not AMENDMENT_9_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 9 absent — the capability-authority "
            "generation does not exist")
    a9 = json.loads(AMENDMENT_9_PATH.read_bytes())
    if a9.get("amends_amendment_8_sha256") != _sha_file(
            AMENDMENT_8_PATH):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 9 does not name amendment 8's "
            "exact bytes")
    if not a9.get("change_disclosure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 9 must disclose its changes")
    if a9.get("proposed_campaign_population"):
        campaign_pins = a9["proposed_campaign_population"]
    a9_pins = a9.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_campaign_orchestrator.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "pipeline_plugins/rl_pipeline_with_validation.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a9_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 9 does not pin the corrected "
                "runtime surface")
    pins.update(a9_pins)
    # C26: amendment 9 is immutable HISTORY — its exact reviewed
    # bytes are pinned; an in-place rewrite refuses even before
    # the link check.
    if _sha_file(AMENDMENT_9_PATH) != AMENDMENT_9_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 9 bytes differ from the reviewed "
            "append-only identity — historical amendments are "
            "never edited in place")
    # C26: amendment 10 — the C23-C25 runtime-authority correction
    # generation; names amendment 9's exact restored bytes,
    # discloses only that correction, and carries the population;
    # its pins supersede.
    if not AMENDMENT_10_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 10 absent — the C23-C25 correction "
            "generation does not exist in the chain")
    a10 = json.loads(AMENDMENT_10_PATH.read_bytes())
    if a10.get("amends_amendment_9_sha256") != AMENDMENT_9_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 10 does not name amendment 9's "
            "exact reviewed bytes")
    if not a10.get("change_disclosure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 10 must disclose its changes")
    if a10.get("scientific_change") != \
            "NONE — runtime authority (C23-C25) only":
        raise B4AuthorityRefusal(
            "REFUSED: amendment 10 must declare NO scientific "
            "change")
    if a10.get("proposed_campaign_population"):
        campaign_pins = a10["proposed_campaign_population"]
    a10_pins = a10.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_campaign_orchestrator.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "pipeline_plugins/rl_pipeline_with_validation.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a10_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 10 does not pin the corrected "
                "runtime surface")
    pins.update(a10_pins)
    if _sha_file(AMENDMENT_10_PATH) != AMENDMENT_10_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 10 bytes differ from the reviewed "
            "append-only identity — historical amendments are "
            "never edited in place")
    # C27: amendment 11 — the finite activation closure. Strict
    # exact schema, self-integral canonical digest; missing,
    # malformed, reordered, transplanted or self-consistently
    # rewritten amendment 11 refuses.
    if not AMENDMENT_11_PATH.is_file():
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 absent — the authorization "
            "consumption is not yet part of the chain")
    a11 = _strict_json_bytes(AMENDMENT_11_PATH.read_bytes(),
                             "amendment 11")
    _A11_KEYS = {"schema", "amends_amendment_10_sha256",
                 "authorization_record_sha256",
                 "owner_ratification_sha256", "order",
                 "change_disclosure", "scientific_change",
                 "proposed_campaign_population",
                 "final_code_pins",
                 "resource_contract_v2_sha256",
                 "chronology_truth", "amendment_sha256"}
    if set(a11) != _A11_KEYS:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 keys are not the exact schema")
    body = {k: a11[k] for k in sorted(a11)
            if k != "amendment_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest() != \
            a11["amendment_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 self-integrity digest does not "
            "re-derive")
    if a11["schema"] != ("agent_multi.b4_superseding_design_"
                         "amendment.v9_activation_closure"):
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 carries a foreign schema")
    if a11["amends_amendment_10_sha256"] != AMENDMENT_10_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 does not name amendment 10's "
            "exact reviewed bytes")
    auth_p = CAMPAIGN_AUTHORIZATION_RECORD_PATH
    if not auth_p.is_file() or _sha_file(auth_p) != \
            a11["authorization_record_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 names an authorization record "
            "whose bytes are absent or differ")
    if a11["owner_ratification_sha256"] != OWNER_RATIFICATION_SHA:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 does not name the owner "
            "ratification's exact bytes")
    if a11["scientific_change"] != \
            "NONE — authorization consumption and C28 portability":
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 must declare NO scientific "
            "change")
    if not a11["change_disclosure"]:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 11 must disclose its changes")
    if a11.get("proposed_campaign_population"):
        campaign_pins = a11["proposed_campaign_population"]
    a11_pins = a11.get("final_code_pins", {})
    for req in ("tools/b4_authority.py", "tools/b4_run_cell.py",
                "tools/b4_campaign_executor.py",
                "tools/b4_campaign_ledger.py",
                "tools/b4_campaign_orchestrator.py",
                "tools/b4_adjudicator.py",
                "tools/materialize_b4_causal_sac.py",
                "pipeline_plugins/rl_pipeline_with_validation.py",
                "tests/test_b4_materializer_authority.py"):
        if req not in a11_pins:
            raise B4AuthorityRefusal(
                "REFUSED: amendment 11 does not pin the complete "
                "final execution, verification and test surface")
    pins.update(a11_pins)
    for rel, want in pins.items():
        live = _sha_file(REPO / rel)
        if live != want:
            raise B4AuthorityRefusal(
                f"REFUSED: executing code {rel} digest {live[:12]} "
                f"differs from the final amendment pin {want[:12]}")
    return {"design_sha256": DESIGN_SHA,
            "amendment_shas": list(AMENDMENT_SHAS)
            + [_sha_file(AMENDMENT_4_PATH),
               _sha_file(AMENDMENT_5_PATH),
               _sha_file(AMENDMENT_6_PATH),
               _sha_file(AMENDMENT_7_PATH),
               _sha_file(AMENDMENT_8_PATH),
               _sha_file(AMENDMENT_9_PATH),
               _sha_file(AMENDMENT_10_PATH),
               _sha_file(AMENDMENT_11_PATH)],
            "final_code_pins": pins,
            "proposed_campaign_population": campaign_pins,
            "design": json.loads(DESIGN_PATH.read_bytes())}


def verify_campaign_materialization(mat_root: Path) -> dict:
    """E8/E9: the campaign tree binds to the amendment-6 PROPOSED
    population identities (auditor-reviewable now; the owner's
    campaign record confirms them later). A self-consistent
    replacement tree grants nothing on this path either."""
    chain = verify_amendment_chain()
    pins = chain["proposed_campaign_population"]
    if not pins:
        raise B4AuthorityRefusal(
            "REFUSED: amendment 6 carries no proposed campaign "
            "population identities — the campaign tree has no "
            "external binding yet")
    mat_root = Path(mat_root)
    for rel, key in (("B4_CELL_CONFIGS.json",
                      "cell_population_sha256"),
                     ("B4_MATERIALIZATION.json",
                      "materialization_sha256"),
                     ("genesis/GENESIS_BINDING.json",
                      "genesis_binding_sha256")):
        want = pins.get(key)
        f = mat_root / rel
        if not want or not f.is_file() or _sha_file(f) != want:
            raise B4AuthorityRefusal(
                f"REFUSED: campaign artifact {rel} differs from the "
                "amendment-6 proposed population identity")
    return chain


# --- E5: evidence-complete comparator verification ----------------
EXPECTED_ARMS = ("B0", "B1", "B2a", "B2b", "B3")
EXPECTED_ORIGINS = (2022, 2023, 2024)
EXPECTED_COST_SET = "alpaca_ethusd"
EXPECTED_CAL_GEOMS = 7
EXPECTED_CAL_ARMS = 4


def verify_comparator_population(baselines_dir: Path,
                                 design: dict,
                                 cost_binding: dict = None) -> dict:
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
        if cost_binding is not None:
            # B4-P3 (order @9fb017e3): the digest is RE-DERIVED from
            # the origin's frozen geometry and the fixed cost bytes —
            # a supplied value is never trusted.
            derived = complete_envelope_digest(
                complete_execution_envelope(
                    frozen[year]["frozen_geometry"], cost_binding),
                cost_binding)
            if r["complete_envelope_digest"] != derived:
                raise B4AuthorityRefusal(
                    f"REFUSED: result {key} complete-envelope digest "
                    "does not re-derive from the frozen geometry and "
                    "fixed cost bytes (B4-P3)")
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
    derived_by_origin = {}
    if cost_binding is not None:
        derived_by_origin = {
            year: complete_envelope_digest(
                complete_execution_envelope(
                    frozen[year]["frozen_geometry"], cost_binding),
                cost_binding)
            for year in EXPECTED_ORIGINS}
    return {"lineage": lineage, "frozen_by_origin": frozen,
            "population_label": packet.get("population_label"),
            "n_results": len(results), "n_ledger": len(rows),
            "complete_envelope_digest_by_origin": derived_by_origin}


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
    # C28: NO operator-specific absolute fallback — the source is
    # one normalized logical relative identity under the accepted
    # predictor root, containment-checked and hashed from the
    # opened descriptor.
    logical_rel = ("examples/data/project3/"
                   "ethusdt_4h_tech_stat_full_model_ready.csv")
    declared = design.get("source_data_path", logical_rel)
    if Path(declared).is_absolute() or ".." in \
            Path(declared).parts:
        raise B4AuthorityRefusal(
            "REFUSED: the sealed design may only name a logical "
            "RELATIVE source identity — absolute or traversing "
            "paths are machine coupling")
    pred_root = resolve_predictor_root().resolve()
    data_p = (pred_root / declared).resolve()
    if pred_root not in data_p.parents:
        raise B4AuthorityRefusal(
            "REFUSED: resolved source escapes the accepted "
            "predictor root")
    import os as _os
    import stat as _stat
    try:
        dfd = _os.open(str(data_p), _os.O_RDONLY | getattr(
            _os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise B4AuthorityRefusal(
            f"REFUSED: source dataset unopenable ({exc})")
    try:
        st = _os.fstat(dfd)
        if not _stat.S_ISREG(st.st_mode):
            raise B4AuthorityRefusal(
                "REFUSED: source dataset is not a regular file")
        h = hashlib.sha256()
        while True:
            chunk = _os.read(dfd, 1 << 20)
            if not chunk:
                break
            h.update(chunk)
    finally:
        _os.close(dfd)
    if h.hexdigest() != design["source_data_sha256"]:
        raise B4AuthorityRefusal(
            "REFUSED: source dataset bytes differ from the sealed "
            "design")
    cost_binding = json.loads(cost_p.read_bytes())[
        "alpaca_ethusd"]["env_binding"]
    comparator = verify_comparator_population(baselines_dir, design,
                                              cost_binding)
    return {"chain": chain, "comparator": comparator,
            "design": design, "cost_binding": cost_binding}
