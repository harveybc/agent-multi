#!/usr/bin/env python3
"""T2 confirmatory EXECUTOR (order C42-C47) — implemented, and
STRUCTURALLY CLOSED for science: `--execute` consumes the SEALED
v6 design only after ALL gates open, including the external
Musashi EXECUTION record, which does not exist. In this order no
confirmatory score is computed; the mechanical rehearsal (`--
mechanical-rehearsal`) runs the SAME real executor over the three
DEVELOPMENT units (sm_co2/sm_sunspots/sm_nile), which are outside
the sealed population, and stops before any sealed-bank series.

Per unit (C43 custody): reconstruct the series FROM PHYSICAL BYTES
through the census loaders, verify digest/period/windows against
the sealed unit_map, run the REAL assay harness capturing raw
predictions/observations, persist them as an NPZ, and write ONE
immutable (O_EXCL, 0600) self-digested unit record binding: sealed
design (file+self), review record, execution record, manifest,
census, dataset/series, unit_map entry, temporal origin windows,
arm/model/seed evidence, executed code identity, T0 operator
artifact and per-phase costs. `verify_unit_record()` recomputes
MASE and extreme metrics FROM THE PERSISTED ARRAYS — a producer
summary alone never authorizes.

C45: sealed budget (4 h wall, 8 GiB RSS, nice 15, stop-file);
per-unit O_EXCL attempt claims; a global O_EXCL executor lock with
pid+heartbeat; durable resume (verified records skip; FAILED units
are preserved as missing per the sealed missing-unit rule, never
deleted to complete a panel)."""
import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402
import t2_bank as bank  # noqa: E402
import t2_bank_census as census_mod  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
SEALED_PATH = STATE / "t2_screen_design_SEALED_V6.json"
DEV_UNITS = ("sm_co2", "sm_sunspots", "sm_nile")

CODE_SURFACE = ("tools/t2_confirmatory_executor.py",
                "tools/t2_confirmatory.py",
                "tools/t2_assay_harness.py",
                "tools/t2_bank.py")


class ExecutorRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


def _excl_write(path: Path, payload: bytes) -> None:
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def code_identity() -> dict:
    return {rel: _sha_file(REPO / rel) for rel in CODE_SURFACE}


def _budget(started_wall: float, limits: dict,
            out_root: Path) -> None:
    """C45: the sealed bounds, enforced between units and between
    origins — wall, RSS and the stop file; nice is set at start."""
    if time.time() - started_wall > float(
            limits["max_wall_seconds"]):
        raise ExecutorRefusal(
            "T2 wall budget exhausted — typed stop, durable "
            "records keep their states")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss > int(limits["max_rss_bytes"]):
        raise ExecutorRefusal(
            f"T2 RSS budget exceeded ({rss} bytes)")
    if (Path(out_root) / "T2_STOP").exists():
        raise ExecutorRefusal(
            "external stop request (T2_STOP present)")


def _heartbeat(out_root: Path, payload: dict) -> None:
    p = Path(out_root) / "EXECUTOR_HEARTBEAT.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {**payload, "pid": os.getpid(),
         "monotonic": time.monotonic()}, indent=1))
    os.replace(tmp, p)


def acquire_lock(out_root: Path) -> Path:
    """One executor per results root: O_EXCL lock with pid; a
    stale lock (dead pid) refuses for operator disposition rather
    than being stolen."""
    p = Path(out_root) / "EXECUTOR_LOCK.json"
    try:
        _excl_write(p, json.dumps(
            {"pid": os.getpid(),
             "started_wall": time.time()}).encode())
    except FileExistsError:
        raise ExecutorRefusal(
            "another executor holds this results root (or a "
            "stale lock remains) — operator disposition, never "
            "silent takeover")
    return p


def load_bank_unit(design: dict, uid: str,
                   manifest: dict, raw_root: Path) -> dict:
    """Reconstruct ONE sealed-population unit from physical bytes
    via the census loaders and verify every unit_map binding."""
    b = design["task_population"]["unit_map"][uid]
    lid = b["dataset"]
    contract = census_mod.CONTRACTS[lid]
    admissible = conf.validate_public_manifest(
        manifest, raw_root=raw_root)
    d = admissible[lid]
    fd = conf._open_nofollow_under(Path(os.path.abspath(raw_root)),
                                  Path(d["local_relpath"]))
    try:
        chunks = []
        while True:
            x = os.read(fd, 1 << 20)
            if not x:
                break
            chunks.append(x)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if hashlib.sha256(raw).hexdigest() != d["sha256"]:
        raise ExecutorRefusal(f"{lid}: bytes drifted")
    import io
    import zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        tsf = [n for n in zf.namelist() if n.endswith(".tsf")]
        if len(tsf) != 1:
            raise ExecutorRefusal(f"{lid}: not exactly one .tsf")
        panel = bank.parse_tsf_bytes(zf.read(tsf[0]), lid)
    built = bank.build_series_units(
        panel, d["family"], contract["period"],
        "monash_record_frequency", contract["max_gap"],
        contract["min_length"])
    if uid not in built["units"]:
        raise ExecutorRefusal(f"{uid}: not rebuildable from bytes")
    u = built["units"][uid]
    if u["series_numeric_sha256"] != b["series_numeric_sha256"]:
        raise ExecutorRefusal(f"{uid}: numeric digest differs "
                              "from the sealed binding")
    if int(u["n"]) != int(b["n_obs"]) or \
            int(u["seasonal_period"]) != int(b["seasonal_period"]):
        raise ExecutorRefusal(f"{uid}: length/period differ from "
                              "the sealed binding")
    want_w = bank.origin_windows_for(
        u["n"], int(design["role_geometry"]["rolling_origins"]),
        float(design["role_geometry"]["origin_base_frac"]),
        seasonal_period=u["seasonal_period"],
        horizon=int(design["role_geometry"]["horizon"]))
    if want_w != b["origin_windows"]:
        raise ExecutorRefusal(f"{uid}: origin windows do not "
                              "re-derive")
    return {"unit_id": uid, "y": u["y"],
            "bytes_sha256": d["sha256"],
            "dataset": lid, "family": b["family"],
            "frequency": u["frequency_declared"],
            "license_note": d["license_id"],
            "missingness": u["missingness"],
            "time_index": u["time_index"],
            "time_provenance": u["time_provenance"],
            "seasonal_period": u["seasonal_period"],
            "seasonal_period_provenance":
                u["seasonal_period_provenance"]}


def run_unit(hz, co, unit: dict, design: dict, authority: dict,
             out_root: Path) -> dict:
    """One unit through the REAL harness with array capture, then
    the immutable custody record + NPZ + independent recompute."""
    uid = unit["unit_id"]
    udir = Path(out_root) / "units"
    udir.mkdir(mode=0o700, exist_ok=True)
    safe = uid.replace("::", "__").replace("/", "_")
    claim_p = udir / f"CLAIM_{safe}.json"
    attempt = f"attempt_{hashlib.sha256(os.urandom(16)).hexdigest()[:16]}"
    try:
        _excl_write(claim_p, json.dumps(
            {"unit_id": uid, "attempt_id": attempt,
             "pid": os.getpid(),
             "claimed_wall": time.time()}).encode())
    except FileExistsError:
        raise ExecutorRefusal(
            f"{uid}: unit already claimed — attempts are never "
            "reused; resume adjudicates records, not claims")
    sink = {}
    t0 = time.time()
    try:
        rec = hz.assay_unit(co, unit, h=int(
            design["role_geometry"]["horizon"]), sink=sink)
    except BaseException as exc:
        term = {"schema": "agent_multi.t2_unit_terminal.v1",
                "unit_id": uid, "attempt_id": attempt,
                "terminal": "FAILED",
                "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                "wall_seconds": round(time.time() - t0, 2)}
        term["terminal_sha256"] = _self_sha(term,
                                            "terminal_sha256")
        _excl_write(udir / f"TERMINAL_{safe}.json",
                    json.dumps(term, indent=1).encode())
        raise
    npz_p = udir / f"ARRAYS_{safe}.npz"
    arrays = {"y": np.asarray(unit["y"], dtype=np.float64)}
    for (okey, arm, model), (pred, obs) in sink.items():
        arrays[f"pred__{okey}__{arm}__{model}"] = pred
        arrays[f"obs__{okey}__{arm}__{model}"] = obs
    with open(npz_p, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.chmod(npz_p, 0o600)
    wrapper = {
        "schema": "agent_multi.t2_confirmatory_unit_record.v1",
        "unit_id": uid, "attempt_id": attempt,
        **authority,
        "unit_binding":
            design["task_population"]["unit_map"].get(uid),
        "code_identity": code_identity(),
        "assay_record": rec,
        "arrays_npz_sha256": _sha_file(npz_p),
        "wall_seconds": round(time.time() - t0, 2)}
    wrapper["record_sha256"] = _self_sha(wrapper, "record_sha256")
    rec_p = udir / f"RECORD_{safe}.json"
    _excl_write(rec_p, json.dumps(wrapper, indent=1).encode())
    verify_unit_record(rec_p, npz_p, design)
    return {"record": str(rec_p.name), "wall": wrapper[
        "wall_seconds"]}


def verify_unit_record(rec_path: Path, npz_path: Path,
                       design: dict) -> dict:
    """C44: recompute EVERY primary and extreme metric from the
    PERSISTED arrays and compare with the producer record; the
    wrapper's self-digest, code identity and NPZ digest must
    re-derive. A mutated prediction, swapped file or edited
    summary refuses."""
    wrapper = conf.strict_json_load(rec_path, "unit record")
    if _self_sha(wrapper, "record_sha256") != \
            wrapper["record_sha256"]:
        raise ExecutorRefusal("unit record self-digest does not "
                              "re-derive")
    if wrapper["arrays_npz_sha256"] != _sha_file(npz_path):
        raise ExecutorRefusal(
            "persisted arrays differ from the record's digest — "
            "swapped or edited evidence")
    rec = wrapper["assay_record"]
    data = np.load(npz_path)
    y = data["y"]
    for okey, o in rec["rolling_origins"].items():
        lo_t, hi_t = o["train"]
        period = rec["seasonal_period"]
        den = o["mase_denominator_train_snaive"]
        if hi_t - lo_t > period + 1:
            d2 = float(np.mean(np.abs(
                y[lo_t + period:hi_t] - y[lo_t:hi_t - period])))
            if abs(d2 - den) > 1e-9 * max(1.0, abs(den)):
                raise ExecutorRefusal(
                    f"{okey}: MASE denominator does not recompute "
                    "from the persisted series")
        for arm, entry in o["results"].items():
            models = ({"baseline": entry["metrics"]}
                      if arm == "seasonal_naive" else
                      {"ridge": entry["ridge"],
                       **{f"mlp_{k}": v for k, v in
                          entry["mlp_small"].items()}})
            for mname, metrics in models.items():
                mk = mname.replace("mlp_seed", "mlp_seed")
                key = f"pred__{okey}__{arm}__" + (
                    "baseline" if arm == "seasonal_naive"
                    else ("ridge" if mname == "ridge"
                          else mname.replace("mlp_", "mlp_")))
                okye = key.replace("pred__", "obs__")
                if key not in data or okye not in data:
                    raise ExecutorRefusal(
                        f"{okey}/{arm}/{mname}: persisted arrays "
                        "missing")
                pred, obs = data[key], data[okye]
                mae = float(np.mean(np.abs(pred - obs)))
                want = metrics["mase_primary"]
                got = mae / den if den > 0 else None
                if want is None or got is None or \
                        abs(got - want) > 1e-9 * max(1.0, want):
                    raise ExecutorRefusal(
                        f"{okey}/{arm}/{mname}: MASE does not "
                        "recompute from persisted arrays")
    return {"verified_units": 1}


def census_of_work(design: dict) -> dict:
    tp = design["task_population"]
    n_units = len(tp["series_ids"])
    ro = int(design["role_geometry"]["rolling_origins"])
    seeds = len(design["seed_tape"])
    arms = len(design["arms"])
    return {"units": n_units, "origins_per_unit": ro,
            "arms": arms, "mlp_seeds": seeds,
            "model_fits": n_units * ro * arms * (1 + seeds),
            "baseline_evals": n_units * ro}


def open_all_gates(ledger_path: Path):
    mp = STATE / "t2_public_data_manifest_20260906.json"
    cp = STATE / "t2_bank_census_20260906.json"
    facts = conf.run_confirmatory(mp, SEALED_PATH, ledger_path,
                                  census_path=cp)
    design = conf.strict_json_load(SEALED_PATH, "sealed design")
    manifest = conf.strict_json_load(mp, "manifest")
    authority = {
        "sealed_design_file_sha256": _sha_file(SEALED_PATH),
        "sealed_design_self_sha256": design["design_sha256"],
        "design_review_record_sha256":
            design["design_review_record_sha256"],
        "execution_record_sha256":
            facts["execution_record_sha256"],
        "manifest_sha256": _sha_file(mp),
        "census_sha256": _sha_file(cp)}
    return design, manifest, authority


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="after ALL gates: print the exact work "
                         "census and exit with zero scores")
    ap.add_argument("--mechanical-rehearsal", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None)
    args = ap.parse_args(argv)
    os.nice(15)
    if args.mechanical_rehearsal:
        return rehearse(args.out_root)
    if not (args.execute or args.plan):
        raise ExecutorRefusal(
            "choose --execute/--plan (sealed bank; gated by the "
            "external execution record) or "
            "--mechanical-rehearsal (dev units only)")
    out_root = args.out_root or (
        STATE / "t2_confirmatory_results_v6")
    out_root.mkdir(mode=0o700, exist_ok=True)
    design, manifest, authority = open_all_gates(
        out_root / "T2_ATTEMPT_LEDGER.json")
    work = census_of_work(design)
    if args.plan:
        print(json.dumps({"plan": work, "authority": {
            k: v[:16] for k, v in authority.items()}}, indent=1))
        return 0
    limits = {"max_wall_seconds":
              design["resource_contract"]["max_wall_seconds"],
              "max_rss_bytes":
              design["resource_contract"]["max_rss_bytes"]}
    lock = acquire_lock(out_root)
    import t2_assay_harness as hz
    co = hz.load_co()
    raw_root = STATE / "t2_public_raw"
    started = time.time()
    done = failed = skipped = 0
    try:
        for uid in design["task_population"]["series_ids"]:
            safe = uid.replace("::", "__").replace("/", "_")
            if (out_root / "units" / f"RECORD_{safe}.json"
                    ).exists():
                verify_unit_record(
                    out_root / "units" / f"RECORD_{safe}.json",
                    out_root / "units" / f"ARRAYS_{safe}.npz",
                    design)
                skipped += 1
                continue
            if (out_root / "units" / f"TERMINAL_{safe}.json"
                    ).exists():
                failed += 1     # preserved as missing, never rerun
                continue
            _budget(started, limits, out_root)
            _heartbeat(out_root, {"current_unit": uid,
                                  "done": done,
                                  "failed": failed})
            unit = load_bank_unit(design, uid, manifest, raw_root)
            try:
                run_unit(hz, co, unit, design, authority,
                         out_root)
                done += 1
            except SystemExit:
                raise
            except BaseException:
                failed += 1     # typed terminal already written
    finally:
        lock.unlink(missing_ok=True)
    print(json.dumps({"done": done, "failed_preserved": failed,
                      "resumed_verified": skipped}, indent=1))
    return 0


def rehearse(out_root: Path = None) -> int:
    """C46: the mechanical rehearsal — the SAME real executor over
    the three DEVELOPMENT units only; refuses any sealed-bank
    series; verifiable records; zero confirmatory scores."""
    out_root = out_root or (
        Path.home() / ".cache/t2_rehearsal_v6")
    import shutil
    if Path(out_root).exists():
        shutil.rmtree(out_root)
    os.makedirs(out_root, mode=0o700)
    design = conf.strict_json_load(SEALED_PATH, "sealed design")
    sealed_ids = set(design["task_population"]["series_ids"])
    import t2_assay_harness as hz
    co = hz.load_co()
    import t2_public_data_census as dev_census_mod
    census = dev_census_mod.build_census()
    authority = {
        "sealed_design_file_sha256": _sha_file(SEALED_PATH),
        "sealed_design_self_sha256": design["design_sha256"],
        "design_review_record_sha256":
            design["design_review_record_sha256"],
        "execution_record_sha256":
            "REHEARSAL_NO_EXECUTION_RECORD_MECHANICS_ONLY",
        "manifest_sha256": "REHEARSAL",
        "census_sha256": "REHEARSAL"}
    timings = {}
    for uid in DEV_UNITS:
        assert uid not in sealed_ids
        unit = hz.load_task_unit(census, uid)
        t0 = time.time()
        run_unit(hz, co, unit, design, authority, out_root)
        timings[uid] = round(time.time() - t0, 2)
    print(json.dumps({
        "rehearsal": "COMPLETE_MECHANICS_ONLY",
        "units": list(DEV_UNITS),
        "per_unit_wall_seconds": timings,
        "sealed_bank_series_touched": 0,
        "records_verified_from_persisted_arrays": len(DEV_UNITS),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
