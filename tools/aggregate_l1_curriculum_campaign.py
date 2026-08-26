"""Aggregate the terminal N / EN-W / EN-F L1 curriculum campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ARMS = ("N", "EN-W", "EN-F")
SEEDS = (101, 202, 303, 404)


class AggregationError(ValueError):
    pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_pair_contract(record: dict, contract_sha: str) -> dict:
    contract = dict(record["contracts"]["pair_contract"])
    if "nested_split_contract_sha256" in contract:
        # v3 (finding P1-316): digest IS the authority — it must match
        # the verified contract content or the record refuses.
        if contract["nested_split_contract_sha256"] != contract_sha:
            raise AggregationError(
                "pair contract pins a different nested-contract digest "
                "than the verified file content")
        return contract
    # legacy v2: absolute path replaced by the verified content digest
    path = Path(str(contract.pop("nested_split_contract", "")))
    if path.name != "eth_nested_split_contract_v1.json":
        raise AggregationError("unexpected nested split contract path")
    contract["nested_split_contract_sha256"] = contract_sha
    return contract


def _verified_state_map(record: dict, manifest_dir: Path,
                        name: str) -> dict:
    """Finding P1-317: the state map must be bound by digest.

    v3 records embed {sha256, named_state_sha256}; the evidence file
    (when present) must hash to the embedded digest and carry the same
    map. Missing, malformed or hash-mismatched evidence REFUSES. Legacy
    v2 records fall back to the evidence file alone."""
    embedded = record.get("selected_manifest")
    path = manifest_dir / f"{name}_selected_manifest.json"
    if embedded is not None:
        emb_map = embedded.get("named_state_sha256")
        emb_sha = embedded.get("sha256")
        if not isinstance(emb_map, dict) or not emb_map or not emb_sha:
            raise AggregationError(
                f"malformed embedded state map for {name}")
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != emb_sha:
                raise AggregationError(
                    f"state-map evidence hash mismatch for {name}")
            file_map = _load(path).get("named_state_sha256") or {}
            if file_map != emb_map:
                raise AggregationError(
                    f"state-map content mismatch for {name}")
        return emb_map
    if record.get("schema") == "agent_multi.l1_curriculum_arm.v3":
        raise AggregationError(
            f"v3 record without embedded state map for {name}")
    if not path.is_file():
        raise AggregationError(f"state map evidence missing for {name}")
    states = _load(path).get("named_state_sha256") or {}
    if not states:
        raise AggregationError(f"state map empty for {name}")
    return states


def _direction(deltas: list[float]) -> str:
    positives = sum(value > 0 for value in deltas)
    negatives = sum(value < 0 for value in deltas)
    median = statistics.median(deltas)
    if positives >= 3 and median > 0:
        return "DIRECTIONAL_SIGNAL_FOR"
    if negatives >= 3 and median < 0:
        return "DIRECTIONAL_SIGNAL_AGAINST"
    return "INCONCLUSIVE"


def aggregate(report_dir: Path, manifest_dir: Path,
              nested_contract: Path) -> dict:
    contract_sha = _sha(nested_contract)
    records: dict[tuple[int, str], dict] = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = report_dir / f"seed{seed}_{arm}_report.json"
            if not path.is_file():
                raise AggregationError(f"missing report: {path.name}")
            record = _load(path)
            if record.get("schema") not in ("agent_multi.l1_curriculum_arm.v2", "agent_multi.l1_curriculum_arm.v3"):
                raise AggregationError(f"foreign schema: {path.name}")
            if record.get("outcome") != "ARM_COMPLETE" or record.get(
                    "normal_accepted") is not True:
                raise AggregationError(f"arm not accepted: {path.name}")
            endpoint = record.get("outer_endpoint") or {}
            if endpoint.get("role") != "outer_validation_2024":
                raise AggregationError(f"wrong outer role: {path.name}")
            if endpoint.get("scored_rows") != 2196:
                raise AggregationError(f"wrong outer rows: {path.name}")
            records[(seed, arm)] = record

    rows = []
    results = {}
    for seed in SEEDS:
        normalized = {
            json.dumps(_normalized_pair_contract(records[(seed, arm)],
                                                 contract_sha),
                       sort_keys=True)
            for arm in ARMS
        }
        if len(normalized) != 1:
            raise AggregationError(f"pair identity mismatch for seed {seed}")
        n_states = _verified_state_map(records[(seed, "N")],
                                       manifest_dir, f"seed{seed}_N")
        n_score = records[(seed, "N")]["outer_endpoint"][
            "primary_score_risk_adjusted_return"]
        for arm in ("EN-W", "EN-F"):
            states = _verified_state_map(records[(seed, arm)],
                                         manifest_dir,
                                         f"seed{seed}_{arm}")
            if set(states) != set(n_states):
                raise AggregationError(
                    f"state map keys differ for seed {seed} {arm}")
            identical = sum(states[key] == value
                            for key, value in n_states.items())
            score = records[(seed, arm)]["outer_endpoint"][
                "primary_score_risk_adjusted_return"]
            rows.append({
                "seed": seed,
                "arm": arm,
                "n_score": n_score,
                "arm_score": score,
                "delta": score - n_score,
                "state_tensors": len(states),
                "identical_state_tensors": identical,
                "easy_treatment_diverged": identical < len(states),
            })

    for arm in ("EN-W", "EN-F"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        deltas = [row["delta"] for row in arm_rows]
        informative = [row for row in arm_rows
                       if row["easy_treatment_diverged"]]
        results[arm] = {
            "deltas": deltas,
            "median_delta": statistics.median(deltas),
            "positive_seeds": sum(value > 0 for value in deltas),
            "negative_seeds": sum(value < 0 for value in deltas),
            "informative_easy_seeds": len(informative),
            "direction_rule_result": _direction(deltas),
            "scientific_disposition": (
                "EASY_TREATMENT_INERT"
                if not informative else "EASY_TREATMENT_OBSERVED"
            ),
        }

    return {
        "schema": "agent_multi.l1_curriculum_terminal_aggregate.v1",
        "reports_complete": len(records),
        "nested_split_contract_sha256": contract_sha,
        "identity_normalization": {
            "field": "nested_split_contract",
            "rule": "absolute path replaced by verified content sha256",
        },
        "rows": rows,
        "results": results,
        "sealed_2025_used": False,
        "promotion_authorized": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--nested-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.reports, args.manifests, args.nested_contract)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["results"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
