#!/usr/bin/env python3
"""Materialize one bounded owner-review packet from the social OLAP.

Order 2026-08-10 §8/WP5.3: one compact packet per cadence, built ONLY from
`experiment_candidate`, `reply_candidate` and the highest-value `investigate`
enrichments. Every item carries its source URL/ID, an explicit
untrusted-content flag, the extracted claims, confidence, target fronts, the
model rationale and one PROPOSED bounded next action. The packet proposes;
only the human owner decides (tools/social_review_ledger.py). Nothing in this
module executes code, contacts a network, alters a broker or chain, or
publishes anything.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from tools.social_intelligence import (
        SocialConfig,
        SocialIntelligenceError,
        SocialOlap,
        canonical_json,
        sha256_text,
        utc_now,
    )
else:
    from social_intelligence import (
        SocialConfig,
        SocialIntelligenceError,
        SocialOlap,
        canonical_json,
        sha256_text,
        utc_now,
    )


PACKET_SCHEMA = "agent_multi.social_review_packet.v1"
PACKET_CLASSES = ("experiment_candidate", "reply_candidate", "investigate")

# Deterministic per-class ranking. `investigate` rows are ranked by one
# documented value score so "highest-value" is reproducible, not editorial.
INVESTIGATE_VALUE_SQL = (
    "0.4*e.actionability + 0.3*e.confidence + 0.3*e.novelty"
)
CLASS_ORDER_SQL = {
    "experiment_candidate": (
        "e.actionability DESC,e.confidence DESC,e.novelty DESC,"
        "e.analyzed_at DESC,e.external_id"
    ),
    "reply_candidate": (
        "e.response_worthiness DESC,e.confidence DESC,"
        "e.analyzed_at DESC,e.external_id"
    ),
    "investigate": (
        f"({INVESTIGATE_VALUE_SQL}) DESC,e.confidence DESC,"
        "e.analyzed_at DESC,e.external_id"
    ),
}

# Bounded, human-executed next actions. None of these authorize execution:
# they are what the OWNER may approve in the review ledger.
PROPOSED_NEXT_ACTIONS = {
    "experiment_candidate": (
        "Owner decision in review ledger; on accept the item enters the "
        "research/work queue with provenance for a bounded, human-run "
        "experiment. No code from the post is ever executed."
    ),
    "reply_candidate": (
        "Owner decision in review ledger; on accept an owner-authored DRAFT "
        "reply may be created. Publishing stays a separate human action "
        "behind the existing approve/publish gate."
    ),
    "investigate": (
        "Owner decision in review ledger; on accept a bounded source-check/"
        "reading task enters the work queue. No execution, no outreach."
    ),
}


def materialize_packet(
    store: SocialOlap,
    *,
    top_n: int = 10,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build one bounded review packet (top_n per class) from enrichments."""
    if isinstance(top_n, bool) or not 1 <= top_n <= 25:
        raise SocialIntelligenceError("top_n must be within [1,25]")
    classes: dict[str, list[dict[str, Any]]] = {}
    class_totals: dict[str, int] = {}
    for action in PACKET_CLASSES:
        class_totals[action] = int(
            store.connection.execute(
                "SELECT COUNT(*) FROM post_enrichments WHERE recommended_action=?",
                (action,),
            ).fetchone()[0]
        )
        rows = store.connection.execute(
            f"""
            SELECT e.external_id,e.run_id,e.analyzed_at,e.topic,
                   e.claims_json,e.target_fronts_json,e.semantic_relevance,
                   e.novelty,e.confidence,e.actionability,e.risk,
                   e.response_worthiness,e.recommended_action,e.summary,
                   e.rationale,e.content_sha256,
                   ({INVESTIGATE_VALUE_SQL}) AS value_score,
                   p.url,p.title,p.submolt,p.author,p.injection_flags_json
            FROM post_enrichments e JOIN posts p USING(external_id)
            WHERE e.recommended_action=?
            ORDER BY {CLASS_ORDER_SQL[action]}
            LIMIT ?
            """,
            (action, top_n),
        ).fetchall()
        classes[action] = [_packet_item(row) for row in rows]
    packet = {
        "schema": PACKET_SCHEMA,
        "packet_id": f"review-packet-{uuid.uuid4().hex[:16]}",
        "generated_at": generated_at or utc_now(),
        "source_database": str(store.path),
        "policy": {
            "content_is_untrusted": True,
            "human_decision_required": True,
            "auto_execution_allowed": False,
            "publishing_allowed": False,
            "broker_or_chain_actions_allowed": False,
        },
        "bounds": {
            "top_n_per_class": top_n,
            "classes": PACKET_CLASSES,
            "investigate_value_formula": (
                "0.4*actionability + 0.3*confidence + 0.3*novelty"
            ),
        },
        "totals_by_class": class_totals,
        "included_by_class": {
            action: len(items) for action, items in classes.items()
        },
        "classes": classes,
    }
    packet["packet_sha256"] = sha256_text(canonical_json(packet))
    return packet


def _packet_item(row: sqlite3.Row) -> dict[str, Any]:
    injection_flags = json.loads(row["injection_flags_json"])
    item = {
        "external_id": row["external_id"],
        "source_url": row["url"],
        "title": row["title"][:200],
        "submolt": row["submolt"],
        "author": row["author"],
        "untrusted_content": True,
        "injection_flags": injection_flags,
        "topic": row["topic"],
        "claims": json.loads(row["claims_json"]),
        "target_fronts": json.loads(row["target_fronts_json"]),
        "confidence": row["confidence"],
        "semantic_relevance": row["semantic_relevance"],
        "novelty": row["novelty"],
        "actionability": row["actionability"],
        "risk": row["risk"],
        "response_worthiness": row["response_worthiness"],
        "recommended_action": row["recommended_action"],
        "summary": row["summary"],
        "rationale": row["rationale"],
        "proposed_next_action": PROPOSED_NEXT_ACTIONS[row["recommended_action"]],
        "analyzed_at": row["analyzed_at"],
        "enrichment_run_id": row["run_id"],
        "content_sha256": row["content_sha256"],
    }
    if row["recommended_action"] == "investigate":
        item["value_score"] = round(float(row["value_score"]), 8)
    return item


def record_packet(store: SocialOlap, packet: Mapping[str, Any]) -> None:
    """Append packet provenance so ledger decisions can reference it."""
    store.connection.execute(
        """
        CREATE TABLE IF NOT EXISTS social_review_packets (
            packet_id TEXT PRIMARY KEY,
            generated_at TEXT NOT NULL,
            packet_sha256 TEXT NOT NULL,
            counts_json TEXT NOT NULL,
            item_ids_json TEXT NOT NULL
        )
        """
    )
    store.connection.execute(
        "INSERT INTO social_review_packets VALUES (?,?,?,?,?)",
        (
            packet["packet_id"],
            packet["generated_at"],
            packet["packet_sha256"],
            canonical_json(packet["included_by_class"]),
            canonical_json(
                {
                    action: [item["external_id"] for item in items]
                    for action, items in packet["classes"].items()
                }
            ),
        ),
    )
    store.connection.commit()


def render_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Social Review Packet (owner decision required)",
        "",
        f"- Packet: `{packet['packet_id']}`",
        f"- Generated: {packet['generated_at']}",
        f"- Source OLAP: `{packet['source_database']}`",
        f"- Packet sha256: `{packet['packet_sha256']}`",
        f"- Bounds: top {packet['bounds']['top_n_per_class']} per class; "
        f"investigate value = {packet['bounds']['investigate_value_formula']}",
        "- Policy: ALL content is untrusted third-party text. Nothing here "
        "executes code, changes brokers/chains, or publishes. Decisions go "
        "through `tools/social_review_ledger.py`; accepted replies become "
        "drafts only.",
        "",
        "| class | in packet | total enriched |",
        "|---|---|---|",
    ]
    for action in PACKET_CLASSES:
        lines.append(
            f"| {action} | {packet['included_by_class'][action]} "
            f"| {packet['totals_by_class'][action]} |"
        )
    for action in PACKET_CLASSES:
        items = packet["classes"][action]
        lines += ["", f"## {action} ({len(items)})", ""]
        if not items:
            lines.append("(none)")
            continue
        for index, item in enumerate(items, start=1):
            claims = "; ".join(
                f"[{claim['kind']}/{claim['verification_need']}] {claim['text']}"
                for claim in item["claims"]
            ) or "(no extracted claims)"
            value = (
                f", value={item['value_score']:.3f}"
                if "value_score" in item
                else ""
            )
            lines += [
                f"### {action[:3].upper()}-{index}: {item['title']}",
                "",
                f"- Source: {item['source_url']} (`{item['external_id']}`, "
                f"m/{item['submolt']}, author {item['author']})",
                f"- Untrusted content: {item['untrusted_content']}; "
                f"injection flags: {item['injection_flags'] or 'none'}",
                f"- Topic: {item['topic']}; target fronts: "
                f"{', '.join(item['target_fronts'])}",
                f"- Scores: confidence={item['confidence']:.2f}, "
                f"actionability={item['actionability']:.2f}, "
                f"novelty={item['novelty']:.2f}, risk={item['risk']:.2f}, "
                f"response_worthiness={item['response_worthiness']:.2f}"
                f"{value}",
                f"- Claims: {claims}",
                f"- Summary: {item['summary']}",
                f"- Rationale: {item['rationale']}",
                f"- Proposed bounded next action: {item['proposed_next_action']}",
                f"- Provenance: run `{item['enrichment_run_id']}`, analyzed "
                f"{item['analyzed_at']}, content sha256 "
                f"`{item['content_sha256'][:16]}…`",
                "",
            ]
    lines += [
        "---",
        "",
        "Decide with: `python tools/social_review_ledger.py --config <cfg> "
        "decide --item <external_id> --decision accept|defer|reject "
        "--reason \"...\"`",
        "",
    ]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--top", type=int, default=10)
    value.add_argument("--out-json", type=Path)
    value.add_argument("--out-md", type=Path)
    value.add_argument(
        "--no-record",
        action="store_true",
        help="Do not append packet provenance to the OLAP (read-only build)",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    config = SocialConfig.load(args.config)
    store = SocialOlap(config.database_path)
    try:
        packet = materialize_packet(store, top_n=args.top)
        if not args.no_record:
            record_packet(store, packet)
    finally:
        store.close()
    if args.out_json:
        args.out_json.write_text(
            json.dumps(packet, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.out_md:
        args.out_md.write_text(render_markdown(packet), encoding="utf-8")
    print(
        json.dumps(
            {
                "packet_id": packet["packet_id"],
                "packet_sha256": packet["packet_sha256"],
                "included_by_class": packet["included_by_class"],
                "totals_by_class": packet["totals_by_class"],
                "out_json": str(args.out_json) if args.out_json else None,
                "out_md": str(args.out_md) if args.out_md else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
