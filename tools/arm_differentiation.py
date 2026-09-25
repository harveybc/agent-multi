#!/usr/bin/env python3
"""Arm differentiation assertion: refuse a campaign whose arms cannot differ.

Precondition for the L2_N vs L2_EN smoke (finding D8-20260815). A
factorial that declares different treatments but MEASURES the same thing
produces a degenerate null and burns the fleet. This module answers one
question per arm pair, with a TYPED verdict:

    "these two arms declare different treatments and report the same
     metric tuple — is that a real outcome, or did the measurement
     pipeline lose the distinction?"

The discriminator is NOT the artifact hash. Two decisive facts from the
eth_curriculum_decision_20260807_v2 post-mortem forbid it:

  * Stable-Baselines3 ``.zip`` containers embed archive timestamps, so
    the SAME weights saved twice yield DIFFERENT container digests. In
    that collection ``E4/model.post_easy.zip`` (ce67e6f4…) and
    ``EN4_10/model.post_easy.zip`` (a3dad941…) are different files whose
    ``policy.pth`` tensors are bit-identical (8137b224…, and equal to
    the warm-start anchor). A container-hash rule calls that a bug; it
    is not a bug, it is a SHARED SCORED POLICY — worse, and different.

  * Conversely, two genuinely distinct policies that both collapse to
    hold trade zero times and therefore MUST report the identical null
    tuple. In p1_difficulty_lr_factorial…_decision/c0e53cf18b7d60dd the
    cells P1E_LR1E4 and P1N_LR1E4 carry distinct policy tensors
    (4312de6d… vs 908c09bb…) and identical all-zero metrics. That
    identity is REAL and expected; flagging it would be a false alarm
    that blocks a correct campaign.

So the ladder below is ordered by what each fact can PROVE, and the two
accept-verdicts are the only ways an identical tuple survives:

  SHARED_SCORED_POLICY  refuse — the arms scored the same weights, so the
                        treatment was never measured at all
  DEGENERATE_IDENTICAL  accept — both arms are degenerate (zero-trade, or a
                        constant action that ignores its observations), so
                        identical metrics are FORCED and therefore real
  TREATMENT_NOT_REALIZED refuse — identical NON-degenerate behaviour under
                        different treatments: a live policy cannot replay
                        the same actions on different inputs, so the
                        treatment never reached the model
  METRIC_COLLAPSE       refuse — distinct policies, distinct behaviour,
                        identical metrics: the measurement lost it
  UNDECIDABLE           refuse — identical tuple with no behavioural
                        evidence; absence of proof is not proof of health

The degeneracy exemption is not a loophole, it is the whole point. A
policy saturated at ``action_raw == 1.0`` for all 28 931 steps (observed
across eight different usdcad feature presets) is BY CONSTRUCTION
independent of its features; demanding that such arms differ would refuse
a correct measurement of a real training collapse. Degeneracy must be
proven from the trace, never assumed — use :func:`trace_behavior_facts`.

``behavior_fingerprint`` must be computed with :func:`trace_fingerprint`,
which hashes a return trace with the identity columns (``seed``,
``run_id``, ``episode_id``) REMOVED — those label columns are the only
difference between two byte-identical rollouts and would otherwise mask
behavioural equality. ``scored_policy_tensor_sha256`` must come from
:func:`policy_tensor_sha256`, which hashes the tensors inside the
container, never the container.

Import-light and socket-free by construction: torch is imported lazily
inside :func:`policy_tensor_sha256` only, so the assertion and its tests
run without it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

REPORT_SCHEMA = "agent_multi.arm_differentiation_report.v1"

#: Verdicts that permit the campaign to proceed.
ACCEPT_VERDICTS = frozenset({
    "OK",
    "SAME_TREATMENT",
    "DEGENERATE_IDENTICAL",
})
#: Verdicts that REFUSE the campaign.
REFUSE_VERDICTS = frozenset({
    "SHARED_SCORED_POLICY",
    "TREATMENT_NOT_REALIZED",
    "METRIC_COLLAPSE",
    "UNDECIDABLE",
    "NO_INFORMATIVE_CONTRAST",
})

#: Columns that label a rollout without describing its behaviour.
IDENTITY_COLUMNS = ("seed", "run_id", "episode_id")

DEFAULT_METRIC_KEYS = ("mean_weekly_return", "annualized_return",
                       "total_return", "max_drawdown_fraction",
                       "trades_total")


class ArmDifferentiationRefusal(RuntimeError):
    """Typed refusal: at least one arm pair failed differentiation."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        self.refusals = [p for p in report.get("pairs", ())
                         if p.get("verdict") in REFUSE_VERDICTS]
        lines = [
            f"{p['arms'][0]} vs {p['arms'][1]}: {p['verdict']}"
            f" — {p['detail']}" for p in self.refusals]
        super().__init__(
            f"{len(self.refusals)} arm pair(s) failed the differentiation"
            " precondition:\n  " + "\n  ".join(lines))


@dataclass(frozen=True)
class ArmObservation:
    """One arm's declared treatment and what was actually measured.

    ``scored_policy_tensor_sha256`` is the hash of the weights that
    produced ``metrics`` — the SELECTED checkpoint, not the terminal
    one, whenever selection and terminal differ. Passing the container
    digest here defeats the whole check.
    """

    arm: str
    treatment: Mapping[str, Any]
    metrics: Mapping[str, Any]
    scored_policy_tensor_sha256: Optional[str] = None
    behavior_fingerprint: Optional[str] = None
    behavior_degenerate: Optional[bool] = None
    active: Optional[bool] = None
    trades_total: Optional[int] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def tuple_for(self, metric_keys: Sequence[str]) -> str:
        """Canonical, exact rendering of the metric tuple.

        ``repr`` on floats round-trips, so two values that merely PRINT
        the same at reduced precision stay distinguishable here.
        """
        parts = []
        for key in metric_keys:
            value = self.metrics.get(key)
            parts.append(f"{key}={value!r}" if not isinstance(value, float)
                         else f"{key}={value.hex()}")
        return "|".join(parts)

    def is_degenerate(self) -> bool:
        """True when this arm's OUTPUT could not have depended on inputs.

        Either the trace proved a constant/observation-independent action
        (``behavior_degenerate``), or the arm never traded, or the run was
        typed inactive. Any of these force identical metrics between
        arms regardless of treatment.
        """
        if self.behavior_degenerate is True:
            return True
        if self.active is False:
            return True
        return self.trades_total == 0


def _treatment_key(treatment: Mapping[str, Any]) -> str:
    return json.dumps(treatment, sort_keys=True, default=str)


def _classify(a: ArmObservation, b: ArmObservation,
              metric_keys: Sequence[str]) -> tuple[str, str]:
    if _treatment_key(a.treatment) == _treatment_key(b.treatment):
        return ("SAME_TREATMENT",
                "arms declare the same treatment; not a comparison")
    if a.tuple_for(metric_keys) != b.tuple_for(metric_keys):
        return ("OK", "metric tuples differ")

    # From here the tuples are byte-identical and the treatments differ.
    ta, tb = a.scored_policy_tensor_sha256, b.scored_policy_tensor_sha256
    if ta is not None and tb is not None and ta == tb:
        return ("SHARED_SCORED_POLICY",
                "both arms scored the identical policy tensor"
                f" {ta[:16]}…; the declared treatment difference was"
                " never measured — identical metrics are arithmetically"
                " correct and scientifically void")

    if a.is_degenerate() and b.is_degenerate():
        return ("DEGENERATE_IDENTICAL",
                "both arms are degenerate (constant action, zero trades"
                " or typed inactive); a policy that ignores its"
                " observations MUST score the same under any treatment,"
                " so this identity is real")

    fa, fb = a.behavior_fingerprint, b.behavior_fingerprint
    if fa is not None and fb is not None:
        if fa == fb:
            return ("TREATMENT_NOT_REALIZED",
                    f"identical behaviour fingerprint {fa[:16]}… from"
                    " distinct policies that are NOT degenerate; a live"
                    " policy cannot replay identical actions under a"
                    " different treatment — the treatment never reached"
                    " the model (check config wiring and the data/feature"
                    " path, not the metric code)")
        return ("METRIC_COLLAPSE",
                f"policies differ ({str(ta)[:12]}… vs {str(tb)[:12]}…)"
                f" and behaviour differs ({fa[:12]}… vs {fb[:12]}…) yet"
                " the metric tuple is byte-identical — the measurement"
                " pipeline lost the distinction")

    return ("UNDECIDABLE",
            "metric tuples are byte-identical and no behaviour"
            " fingerprint was supplied, so legitimacy cannot be"
            " established; supply trace_behavior_facts() for both arms")


def evaluate_arms(observations: Iterable[ArmObservation], *,
                  metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
                  ) -> dict:
    """Classify every unordered arm pair. Never raises on a verdict."""
    arms = list(observations)
    seen: set[str] = set()
    for obs in arms:
        if obs.arm in seen:
            raise ValueError(f"duplicate arm label {obs.arm!r}")
        seen.add(obs.arm)

    pairs = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            verdict, detail = _classify(a, b, metric_keys)
            pairs.append({
                "arms": [a.arm, b.arm],
                "verdict": verdict,
                "detail": detail,
                "metric_tuple_identical": (
                    a.tuple_for(metric_keys) == b.tuple_for(metric_keys)),
                "scored_policy_tensor_sha256": [
                    a.scored_policy_tensor_sha256,
                    b.scored_policy_tensor_sha256],
                "behavior_fingerprint": [
                    a.behavior_fingerprint, b.behavior_fingerprint],
                "behavior_degenerate": [
                    a.is_degenerate(), b.is_degenerate()],
            })
    refused = [p for p in pairs if p["verdict"] in REFUSE_VERDICTS]
    contrasts = [p for p in pairs if p["verdict"] != "SAME_TREATMENT"]
    informative = any(p["verdict"] == "OK" for p in contrasts)
    return {
        "schema": REPORT_SCHEMA,
        "metric_keys": list(metric_keys),
        "arms": [o.arm for o in arms],
        "pairs": pairs,
        "refused": len(refused),
        "differentiated": not refused,
        # A campaign can be free of measurement defects and still be
        # unable to answer its question: every cross-treatment pair
        # degenerate means the arms all collapsed. That is a real
        # outcome, not a bug, so it is reported — not refused by
        # default. Opt in with require_informative=True.
        "contrast_pairs": len(contrasts),
        "degenerate_identical": sum(
            1 for p in contrasts if p["verdict"] == "DEGENERATE_IDENTICAL"),
        "informative": informative,
    }


def assert_arms_differentiated(
        observations: Iterable[ArmObservation], *,
        metric_keys: Sequence[str] = DEFAULT_METRIC_KEYS,
        require_informative: bool = False) -> dict:
    """Return the report, or raise :class:`ArmDifferentiationRefusal`.

    This is the L2 smoke precondition. Call it BEFORE dispatching GPUs.

    With ``require_informative=True`` a run in which no cross-treatment
    pair separates at all (every arm degenerate) is ALSO refused — the
    measurement is sound but the campaign cannot answer its question.
    """
    report = evaluate_arms(observations, metric_keys=metric_keys)
    failed = not report["differentiated"]
    if require_informative and report["contrast_pairs"] and not report[
            "informative"]:
        failed = True
        report = dict(report)
        report["pairs"] = list(report["pairs"]) + [{
            "arms": ["*", "*"],
            "verdict": "NO_INFORMATIVE_CONTRAST",
            "detail": ("no cross-treatment pair separated; every arm is"
                       " degenerate, so this campaign cannot answer its"
                       " question (measurement is sound)"),
            "metric_tuple_identical": True,
            "scored_policy_tensor_sha256": [None, None],
            "behavior_fingerprint": [None, None],
            "behavior_degenerate": [True, True],
        }]
        report["differentiated"] = False
    if failed:
        raise ArmDifferentiationRefusal(report)
    return report


# --------------------------------------------------------------------
# Fingerprint helpers — the evidence the ladder above depends on.
# --------------------------------------------------------------------

def trace_fingerprint(path: Path | str, *,
                      drop_columns: Sequence[str] = IDENTITY_COLUMNS,
                      ) -> str:
    """Hash a return trace's BEHAVIOUR, ignoring identity label columns.

    ``seed``/``run_id``/``episode_id`` differ between replicas of the
    same rollout and would otherwise make two byte-identical behaviours
    look distinct.
    """
    digest = hashlib.sha256()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: return trace has no header")
        columns = [c for c in reader.fieldnames if c not in drop_columns]
        digest.update("\x1e".join(columns).encode())
        for row in reader:
            digest.update("\x1f".join(
                str(row.get(c, "")) for c in columns).encode())
            digest.update(b"\x1e")
    return digest.hexdigest()


def trace_behavior_facts(path: Path | str, *,
                         action_column: str = "action_raw",
                         trades_column: str = "trades",
                         drop_columns: Sequence[str] = IDENTITY_COLUMNS,
                         ) -> dict:
    """Fingerprint a return trace AND decide whether it is degenerate.

    Degenerate means the action stream carried no information: a single
    distinct action value over the whole episode (a saturated or
    hold-collapsed policy), or no trades at all. Such a policy's output
    cannot depend on its observations, so two arms sharing this property
    are ENTITLED to identical metrics.
    """
    digest = hashlib.sha256()
    actions: set[str] = set()
    max_trades = 0
    rows = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: return trace has no header")
        columns = [c for c in reader.fieldnames if c not in drop_columns]
        digest.update("\x1e".join(columns).encode())
        for row in reader:
            rows += 1
            digest.update("\x1f".join(
                str(row.get(c, "")) for c in columns).encode())
            digest.update(b"\x1e")
            if action_column in row and row[action_column] != "":
                actions.add(str(row[action_column]))
            raw = row.get(trades_column)
            if raw not in (None, ""):
                try:
                    max_trades = max(max_trades, int(float(raw)))
                except ValueError:
                    pass
    return {
        "behavior_fingerprint": digest.hexdigest(),
        "rows": rows,
        "distinct_actions": len(actions),
        "trades_total": max_trades,
        "behavior_degenerate": len(actions) <= 1 or max_trades == 0,
    }


def policy_tensor_sha256(path: Path | str, *,
                         member: str = "policy.pth") -> str:
    """Hash the TENSORS inside a Stable-Baselines3 container.

    The container digest is unusable as an identity: ``.zip`` archives
    embed member timestamps, so identical weights saved twice hash
    differently. This reads ``policy.pth`` and hashes name, dtype, shape
    and raw bytes of every tensor in sorted key order.
    """
    import torch  # lazy: keeps the assertion importable without torch

    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise ValueError(
                f"{path}: no {member!r} member (found"
                f" {sorted(archive.namelist())})")
        payload = archive.read(member)
    state = torch.load(io.BytesIO(payload), map_location="cpu",
                       weights_only=False)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    if not isinstance(state, dict):
        raise ValueError(f"{path}: {member!r} is not a state dict")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def observations_from_json(payload: Sequence[Mapping[str, Any]],
                           ) -> list[ArmObservation]:
    """Build observations from a plain JSON list (the CLI's input)."""
    out = []
    for entry in payload:
        missing = {"arm", "treatment", "metrics"} - set(entry)
        if missing:
            raise ValueError(
                f"arm entry missing required field(s) {sorted(missing)}")
        out.append(ArmObservation(
            arm=str(entry["arm"]),
            treatment=entry["treatment"],
            metrics=entry["metrics"],
            scored_policy_tensor_sha256=entry.get(
                "scored_policy_tensor_sha256"),
            behavior_fingerprint=entry.get("behavior_fingerprint"),
            behavior_degenerate=entry.get("behavior_degenerate"),
            active=entry.get("active"),
            trades_total=entry.get("trades_total"),
            provenance=entry.get("provenance") or {},
        ))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arms-json", type=Path, required=True,
        help="JSON list of arm observations (see observations_from_json)")
    parser.add_argument("--metric-keys", default=",".join(DEFAULT_METRIC_KEYS))
    parser.add_argument("--report", type=Path, default=None,
                        help="write the typed report here")
    parser.add_argument(
        "--require-informative", action="store_true",
        help=("also refuse when every cross-treatment pair is degenerate"
              " (sound measurement, unanswerable campaign)"))
    args = parser.parse_args(argv)

    payload = json.loads(args.arms_json.read_text(encoding="utf-8"))
    observations = observations_from_json(payload)
    metric_keys = tuple(k for k in args.metric_keys.split(",") if k)
    try:
        report = assert_arms_differentiated(
            observations, metric_keys=metric_keys,
            require_informative=args.require_informative)
        refused = False
    except ArmDifferentiationRefusal as exc:
        report, refused = exc.report, True
    text = json.dumps(report, indent=1, default=str) + "\n"
    if args.report is not None:
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    if refused:
        print("REFUSED: arms are not differentiated — do NOT dispatch",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
