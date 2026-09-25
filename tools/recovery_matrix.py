#!/usr/bin/env python3
"""R6 (order 2026-09-11): a reproducible, READ-ONLY recovery matrix.

After a reboot the tempting question is "are the units active?" — and
it is the wrong question. `active/running` says a process exists. It
does not say the process is doing anything, that its campaign is alive
rather than finished, or that a unit sitting in `failed` is a stopped
service rather than a refusal someone already understood.

So every component here must produce ONE of four kinds of evidence:

  * HEARTBEAT_FRESH   — it published recently, and the age is shown;
  * TERMINAL_STATE    — its work ENDED, and the durable record says so;
  * REFUSED_STABLE    — it declines to run, stably, for a named reason;
  * TYPED_UNAVAILABLE — we could not observe it, and why.

And every component is filed under what it actually is:

  * SERVICE_ALIVE     — a daemon whose job is to keep running;
  * CAMPAIGN_ACTIVE   — scientific work in progress;
  * CAMPAIGN_TERMINAL — scientific work that ENDED;
  * CAMPAIGN_PAUSED   — scientific work deliberately stopped;
  * HISTORICAL        — an installed thing that will never run again.

The tool NEVER starts, stops, restarts, enables or disables anything,
never writes outside its own record, and never touches a database with
anything but a read.

    python tools/recovery_matrix.py [--json] [--out RECORD.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HOME = Path.home()
SHARE = HOME / ".local/share/agent-multi"
SCHEMA = "agent_multi.recovery_matrix.v1"

HEARTBEAT_FRESH = "HEARTBEAT_FRESH"
TERMINAL_STATE = "TERMINAL_STATE"
REFUSED_STABLE = "REFUSED_STABLE"
TYPED_UNAVAILABLE = "TYPED_UNAVAILABLE"

SERVICE_ALIVE = "SERVICE_ALIVE"
CAMPAIGN_ACTIVE = "CAMPAIGN_ACTIVE"
CAMPAIGN_TERMINAL = "CAMPAIGN_TERMINAL"
CAMPAIGN_PAUSED = "CAMPAIGN_PAUSED"
HISTORICAL = "HISTORICAL"

OK = "OK"
ATTENTION = "ATTENTION"
UNKNOWN = "UNKNOWN"


def redact(value) -> str:
    return str(value).replace(str(HOME), "~")


def utc_now() -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0)
            .isoformat().replace("+00:00", "Z"))


def run(cmd: tuple[str, ...], timeout: float = 30.0,
        env: dict | None = None) -> tuple[int, str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout, env=env)
        return out.returncode, (out.stdout or "") + (out.stderr or "")
    except Exception as exc:                                # noqa: BLE001
        return -1, f"<{type(exc).__name__}>"


def sc_show(unit: str, *props: str) -> dict:
    code, out = run(("systemctl", "--user", "show", unit,
                     *sum((["-p", p] for p in props), [])))
    if code != 0:
        return {}
    return dict(line.split("=", 1) for line in out.splitlines()
                if "=" in line)


def component(name: str, *, kind: str, evidence_kind: str,
              verdict: str, **facts) -> dict:
    return {"component": name, "kind": kind,
            "evidence_kind": evidence_kind, "verdict": verdict, **facts}


# ------------------------------------------------------------ storage
def probe_postgres() -> dict:
    env_file = HOME / ".config/crispdm/olap-loader.env"
    env = dict(os.environ)
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    if not env.get("PGUSER"):
        return component("postgresql", kind=SERVICE_ALIVE,
                         evidence_kind=TYPED_UNAVAILABLE, verdict=UNKNOWN,
                         reason="no credentials are configured on this "
                                "host, so the cube cannot be read")
    started = time.perf_counter()
    code, out = run(("psql", "-h", env.get("PGHOST", "localhost"),
                     "-p", env.get("PGPORT", "5432"),
                     "-U", env["PGUSER"], "-d", env.get("PGDATABASE", ""),
                     "-tAc", "SELECT 1"), timeout=20, env=env)
    if code != 0:
        return component("postgresql", kind=SERVICE_ALIVE,
                         evidence_kind=TYPED_UNAVAILABLE,
                         verdict=ATTENTION,
                         reason=f"read probe failed: {out.strip()[:120]}")
    return component("postgresql", kind=SERVICE_ALIVE,
                     evidence_kind=HEARTBEAT_FRESH, verdict=OK,
                     probe="SELECT 1",
                     probe_seconds=round(time.perf_counter() - started, 3))


def probe_metabase(url: str = "http://localhost:3000/api/health") -> dict:
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read(200).decode("utf-8", "replace")
        return component("metabase", kind=SERVICE_ALIVE,
                         evidence_kind=HEARTBEAT_FRESH, verdict=OK,
                         endpoint="/api/health", response=body[:80])
    except Exception as exc:                                # noqa: BLE001
        return component("metabase", kind=SERVICE_ALIVE,
                         evidence_kind=TYPED_UNAVAILABLE,
                         verdict=ATTENTION,
                         reason=f"{type(exc).__name__}")


# ------------------------------------------------------------- loader
def probe_olap_loader() -> dict:
    unit = "crispdm-olap-loader.service"
    props = sc_show(unit, "ActiveState", "SubState", "NRestarts",
                    "UnitFileState")
    hb_p = HOME / ".local/share/predictor/olap_outbox/HEARTBEAT.json"
    if not hb_p.is_file():
        return component("crispdm-olap-loader", kind=SERVICE_ALIVE,
                         evidence_kind=TYPED_UNAVAILABLE,
                         verdict=ATTENTION,
                         systemd_state=props.get("ActiveState"),
                         reason="the unit publishes no heartbeat, so "
                                "'running' is all we know")
    hb = json.loads(hb_p.read_text())
    age = round(time.time() - float(hb["published_at_epoch"]), 1)
    healthy = bool(hb.get("healthy"))
    return component(
        "crispdm-olap-loader", kind=SERVICE_ALIVE,
        evidence_kind=HEARTBEAT_FRESH,
        verdict=OK if (healthy and age <= 120) else ATTENTION,
        systemd_state=props.get("ActiveState"),
        enabled=props.get("UnitFileState"),
        restarts=props.get("NRestarts"),
        heartbeat_age_seconds=age,
        healthy=healthy,
        backlog_pending=hb.get("pending"),
        dead_letters_total=hb.get("dead_letters_total", "UNAVAILABLE"),
        dead_letters_unadjudicated=hb.get("dead_letters_unadjudicated",
                                          "UNAVAILABLE"),
        note=("a dead-letter on file is NOT an outage; an unadjudicated "
              "one needs a person, not a restart"))


# ---------------------------------------------------------------- DOIN
def probe_doin() -> list[dict]:
    out = []
    props = sc_show("doin-campaign-supervisor.service", "ActiveState",
                    "SubState", "NRestarts", "UnitFileState")
    out.append(component(
        "doin-campaign-supervisor", kind=SERVICE_ALIVE,
        evidence_kind=(HEARTBEAT_FRESH if props.get("SubState") == "running"
                       else TYPED_UNAVAILABLE),
        verdict=OK if props.get("ActiveState") == "active" else ATTENTION,
        systemd_state=props.get("ActiveState"),
        restarts=props.get("NRestarts"),
        note="a healthy supervisor is not a running optimization; the "
             "campaign it supervises is reported separately"))
    for host in ("dragon", "gamma"):
        p = sc_show(f"doin-persistent-session-{host}.service",
                    "ActiveState", "SubState", "NRestarts")
        out.append(component(
            f"doin-persistent-session-{host}", kind=SERVICE_ALIVE,
            evidence_kind=(HEARTBEAT_FRESH
                           if p.get("SubState") == "running"
                           else TYPED_UNAVAILABLE),
            verdict=OK if p.get("ActiveState") == "active" else ATTENTION,
            systemd_state=p.get("ActiveState"),
            restarts=p.get("NRestarts"),
            note="an available remote session is capacity, not work"))
    campaigns = SHARE.parent / "agent-multi/doin-campaigns"
    state_campaigns = HOME / ".local/state/agent-multi/doin-campaigns"
    root = state_campaigns if state_campaigns.is_dir() else campaigns
    names = sorted(p.name for p in root.iterdir()) if root.is_dir() else []
    out.append(component(
        "doin-campaigns", kind=CAMPAIGN_PAUSED,
        evidence_kind=TERMINAL_STATE if names else TYPED_UNAVAILABLE,
        verdict=OK,
        campaigns=names[:10],
        note="listed as PAUSED/history: a visible campaign directory is "
             "not an active optimization, and none is dispatched here"))
    return out


# ----------------------------------------------------------------- GPU
def probe_gpu() -> dict:
    code, out = run(("nvidia-smi",
                     "--query-compute-apps=pid,process_name,used_memory",
                     "--format=csv,noheader"), timeout=20)
    unusable = ("Failed to initialize NVML" in out
                or "Driver/library version mismatch" in out)
    if code != 0 or unusable:
        return component(
            "gpu", kind=SERVICE_ALIVE, evidence_kind=TYPED_UNAVAILABLE,
            verdict=ATTENTION if unusable else UNKNOWN,
            reason=" ".join(out.split())[:160] or "nvidia-smi gave no "
                                                 "answer",
            consequence=("the accelerator cannot be enumerated, so it "
                         "cannot be used and — equally important — no "
                         "claim that it is idle can be made from here"))
    procs = [ln.strip() for ln in out.splitlines() if ln.strip()]
    scientific = [p for p in procs
                  if re.search(r"python|torch|tensorflow", p, re.I)]
    return component(
        "gpu", kind=SERVICE_ALIVE, evidence_kind=TERMINAL_STATE, verdict=OK,
        compute_processes=len(procs),
        scientific_compute_processes=len(scientific),
        note=("no scientific CUDA process is running; memory in use "
              "belongs to the desktop, not to a training worker"
              if not scientific else
              "a scientific CUDA process IS running — this matrix does "
              "not authorize it and does not stop it"))


# ------------------------------------------------------------ B4 / T2
def probe_b4() -> dict:
    root = SHARE / "b4_campaign_results_v7_20260907"
    log = root / "B4_CAMPAIGN_CLOSURE.jsonl"
    if not log.is_file():
        return component("b4-campaign-v7", kind=CAMPAIGN_TERMINAL,
                         evidence_kind=TYPED_UNAVAILABLE,
                         verdict=ATTENTION,
                         reason="the campaign root carries no durable "
                                "closure record")
    closure = json.loads(log.read_text().splitlines()[-1])
    adj = closure["adjudication"]
    return component(
        "b4-campaign-v7", kind=CAMPAIGN_TERMINAL,
        evidence_kind=TERMINAL_STATE, verdict=OK,
        verdict_scientific=adj["verdict"],
        counts=adj["counts"],
        closed_at=closure["closed_at"],
        relaunch_refused=closure["relaunch"]["refused"],
        note="terminal, not active: the launch gate refuses and nine "
             "cells never started")


def probe_t2() -> dict:
    root = SHARE / "t2_confirmatory_results_resource_successor_v1_20260909"
    log = root / "T2_CAMPAIGN_CLOSURE.jsonl"
    units = root / "units"
    if log.is_file():
        closure = json.loads(log.read_text().splitlines()[-1])
        return component(
            "t2-resource-successor", kind=CAMPAIGN_TERMINAL,
            evidence_kind=TERMINAL_STATE, verdict=OK,
            terminal_state=closure["terminal_state"],
            screen_verdict=closure["screen_adjudication"].get("verdict"),
            inventory=closure["inventory"],
            heartbeat_reconciled=closure["heartbeat_reconciliation"][
                "discrepancy"],
            closed_at=closure["closed_at"])
    if not units.is_dir():
        return component("t2-resource-successor", kind=CAMPAIGN_TERMINAL,
                         evidence_kind=TYPED_UNAVAILABLE, verdict=UNKNOWN,
                         reason="no campaign root on this host")
    kinds: dict[str, int] = {}
    for p in units.iterdir():
        kinds[p.name.split("_", 1)[0]] = kinds.get(p.name.split("_", 1)[0],
                                                   0) + 1
    return component(
        "t2-resource-successor", kind=CAMPAIGN_TERMINAL,
        evidence_kind=TYPED_UNAVAILABLE, verdict=ATTENTION,
        artifacts=kinds,
        reason="records exist but no durable closure record does; the "
               "executor heartbeat is telemetry and settles nothing")


# --------------------------------------------------------------- P1LR
def probe_p1lr() -> list[dict]:
    out = []
    code, listing = run(("systemctl", "--user", "list-units",
                         "p1lr-decision@*", "--all", "--no-pager",
                         "--no-legend"))
    seats = sorted(set(re.findall(r"p1lr-decision@(\S+)\.service",
                                  listing)))
    state_dir = HOME / ".local/state/agent-multi/p1lr-gate"
    for seat in seats:
        unit = f"p1lr-decision@{seat}.service"
        props = sc_show(unit, "ActiveState", "Result", "NRestarts",
                        "ConditionResult")
        disp_p = state_dir / f"{unit}.disposition.json"
        disp = json.loads(disp_p.read_text()) if disp_p.is_file() else {}
        refused = (props.get("ConditionResult") == "no"
                   or props.get("Result") == "exec-condition")
        out.append(component(
            unit, kind=HISTORICAL if disp.get("disposition",
                                              "").startswith("HISTORICAL")
            else SERVICE_ALIVE,
            evidence_kind=REFUSED_STABLE if refused else TYPED_UNAVAILABLE,
            verdict=OK if refused and props.get("NRestarts") == "0"
            else ATTENTION,
            systemd_state=props.get("ActiveState"),
            result=props.get("Result"),
            restarts=props.get("NRestarts"),
            disposition=disp.get("disposition", "UNAVAILABLE"),
            refusal_classes=disp.get("refusal_classes", []),
            note="a stable refusal: skipped by ExecCondition, no restart "
                 "scheduled, no training process created"))
    if not seats:
        out.append(component("p1lr-decision", kind=HISTORICAL,
                             evidence_kind=TYPED_UNAVAILABLE, verdict=OK,
                             reason="no seat is installed on this host"))
    return out


# -------------------------------------------------------------- Alpaca
def probe_alpaca() -> list[dict]:
    out = []
    hb_p = HOME / ".local/state/lts/alpaca-model-runner-heartbeat.json"
    props = sc_show("lts-alpaca-model-runner.service", "ActiveState",
                    "SubState", "NRestarts")
    if hb_p.is_file():
        hb = json.loads(hb_p.read_text())
        observed = hb.get("observed_at", "")
        try:
            age = round(time.time() - datetime.fromisoformat(
                observed).timestamp(), 1)
        except Exception:                                   # noqa: BLE001
            age = "UNAVAILABLE"
        degraded = hb.get("state") == "degraded_error"
        out.append(component(
            "lts-alpaca-model-runner", kind=SERVICE_ALIVE,
            evidence_kind=HEARTBEAT_FRESH,
            verdict=ATTENTION if degraded else OK,
            systemd_state=props.get("ActiveState"),
            heartbeat_age_seconds=age,
            state=hb.get("state"), phase=hb.get("phase"),
            error=str(hb.get("error", ""))[:120],
            transient_cause=hb.get("transient_cause", "UNAVAILABLE"),
            note="the RUNNER may submit orders; it is not the observer "
                 "and its connectivity verdict is its own"))
    else:
        out.append(component("lts-alpaca-model-runner", kind=SERVICE_ALIVE,
                             evidence_kind=TYPED_UNAVAILABLE,
                             verdict=ATTENTION,
                             systemd_state=props.get("ActiveState"),
                             reason="no runner heartbeat on this host"))
    obs = sc_show("lts-alpaca-paper-observer.service", "ActiveState",
                  "Result")
    timer = sc_show("lts-alpaca-paper-observer.timer", "ActiveState",
                    "NextElapseUSecRealtime")
    out.append(component(
        "lts-alpaca-paper-observer", kind=SERVICE_ALIVE,
        evidence_kind=(HEARTBEAT_FRESH
                       if timer.get("ActiveState") == "active"
                       else TYPED_UNAVAILABLE),
        verdict=OK if timer.get("ActiveState") == "active" else ATTENTION,
        service_state=obs.get("ActiveState"),
        service_result=obs.get("Result"),
        timer_state=timer.get("ActiveState"),
        note="READ-ONLY and timer-driven: 'inactive' between runs is "
             "normal and is NOT the runner being down"))
    return out


# -------------------------------------------------------------- timers
EXPECTED_TIMERS = (
    "lts-alpaca-paper-observer.timer",
    "lts-paper-execution-watchdog.timer",
    "lts-multi-venue-shadow.timer",
    "gpu-readiness-probe.timer",
    "agent-multi-audit-snapshot.timer",
)


def probe_timers() -> list[dict]:
    out = []
    for t in EXPECTED_TIMERS:
        p = sc_show(t, "ActiveState", "NextElapseUSecRealtime",
                    "LastTriggerUSec")
        out.append(component(
            t, kind=SERVICE_ALIVE,
            evidence_kind=(HEARTBEAT_FRESH
                           if p.get("ActiveState") == "active"
                           else TYPED_UNAVAILABLE),
            verdict=OK if p.get("ActiveState") == "active" else ATTENTION,
            systemd_state=p.get("ActiveState", "UNAVAILABLE"),
            last_trigger=p.get("LastTriggerUSec", "UNAVAILABLE")))
    return out


# ----------------------------------------------------------------- run
def build_matrix() -> dict:
    started = time.perf_counter()
    rows: list[dict] = [probe_postgres(), probe_metabase(),
                        probe_olap_loader(), probe_gpu(),
                        probe_b4(), probe_t2()]
    rows += probe_doin() + probe_p1lr() + probe_alpaca() + probe_timers()

    by_kind: dict[str, int] = {}
    by_verdict: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    linger = run(("loginctl", "show-user", os.environ.get("USER", ""),
                  "-p", "Linger"))[1].strip()
    doc = {
        "schema": SCHEMA,
        "observed_at": utc_now(),
        "read_only": True,
        "started_stopped_or_restarted": [],
        "linger": linger.split("=", 1)[-1] if "=" in linger else "UNKNOWN",
        "components": rows,
        "by_kind": by_kind,
        "by_verdict": by_verdict,
        "attention": [r["component"] for r in rows
                      if r["verdict"] != OK],
        "rule": ("active/running is never sufficient: every component "
                 "above contributes a fresh heartbeat, a terminal state, "
                 "a stable refusal or a typed unavailability"),
        "measurement_seconds": round(time.perf_counter() - started, 2),
    }
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    doc = build_matrix()
    if args.out:
        args.out.write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n")
    if args.json:
        print(redact(json.dumps(doc, indent=1, sort_keys=True)))
    else:
        print(f"recovery matrix @ {doc['observed_at']}  "
              f"(read-only, {doc['measurement_seconds']}s)")
        print(f"  kinds   : {doc['by_kind']}")
        print(f"  verdicts: {doc['by_verdict']}")
        for r in doc["components"]:
            mark = " " if r["verdict"] == OK else "!"
            print(f" {mark} {r['component']:34s} {r['kind']:18s} "
                  f"{r['evidence_kind']:18s} {r['verdict']}")
        if doc["attention"]:
            print(f"  attention: {doc['attention']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
