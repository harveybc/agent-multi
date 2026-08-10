#!/usr/bin/env python3
"""Append-only owner-review ledger for social intelligence candidates.

Order 2026-08-10 §8/WP5.4. The HUMAN OWNER records `accept`, `defer` or
`reject` (with a mandatory reason) against enriched items surfaced by
tools/social_review_packet.py. Semantics:

- the decision journal is append-only, enforced by SQLite triggers; a wrong
  decision is corrected by appending a new one, never by rewriting history;
- an accepted `experiment_candidate` (or `investigate`) enters the
  research/work queue exactly once, with full provenance and a collision
  check on both external_id and content hash;
- an accepted `reply_candidate` may at most create a DRAFT through the
  existing drafts table; publishing remains the separate, already
  human-gated approve/publish path in tools/social_intelligence.py;
- nothing in this module executes code, contacts a network, alters a broker
  or chain, or publishes. This tool records human decisions; it never makes
  them, and it must never be pre-populated by an agent.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from tools.social_intelligence import (
        SocialConfig,
        SocialIntelligenceError,
        SocialOlap,
        canonical_json,
        utc_now,
    )
else:
    from social_intelligence import (
        SocialConfig,
        SocialIntelligenceError,
        SocialOlap,
        canonical_json,
        utc_now,
    )


LEDGER_SCHEMA = "agent_multi.social_review_ledger.v1"
DECISIONS = ("accept", "defer", "reject")
QUEUE_KIND_BY_ACTION = {
    "experiment_candidate": "experiment",
    "investigate": "investigation",
}


class ReviewLedger:
    """Owner-decision ledger and research/work queue over the social OLAP."""

    def __init__(self, store: SocialOlap) -> None:
        self.store = store
        self.connection = store.connection
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS owner_review_decisions (
                decision_id TEXT PRIMARY KEY,
                decided_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                external_id TEXT NOT NULL REFERENCES posts(external_id),
                packet_id TEXT,
                decision TEXT NOT NULL
                    CHECK (decision IN ('accept','defer','reject')),
                reason TEXT NOT NULL CHECK (length(reason) > 0),
                recommended_action TEXT NOT NULL,
                enrichment_run_id TEXT NOT NULL,
                content_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS owner_review_decisions_item_idx
                ON owner_review_decisions(external_id,decided_at DESC);
            CREATE TRIGGER IF NOT EXISTS owner_review_decisions_no_update
            BEFORE UPDATE ON owner_review_decisions BEGIN
                SELECT RAISE(ABORT,'owner_review_decisions is append-only');
            END;
            CREATE TRIGGER IF NOT EXISTS owner_review_decisions_no_delete
            BEFORE DELETE ON owner_review_decisions BEGIN
                SELECT RAISE(ABORT,'owner_review_decisions is append-only');
            END;
            CREATE TABLE IF NOT EXISTS social_work_queue (
                queue_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                external_id TEXT NOT NULL UNIQUE,
                decision_id TEXT NOT NULL
                    REFERENCES owner_review_decisions(decision_id),
                title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                claims_json TEXT NOT NULL,
                target_fronts_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'queued'
            );
            """
        )
        self.connection.commit()

    def _enriched_item(self, external_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            """
            SELECT e.external_id,e.run_id,e.recommended_action,e.claims_json,
                   e.target_fronts_json,e.content_sha256,e.analyzed_at,
                   p.title,p.url,p.submolt
            FROM post_enrichments e JOIN posts p USING(external_id)
            WHERE e.external_id=?
            """,
            (external_id,),
        ).fetchone()
        if row is None:
            raise SocialIntelligenceError(
                f"Item {external_id} is not in the enrichment OLAP; "
                "only enriched items can be decided"
            )
        return row

    def latest_decision(self, external_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT * FROM owner_review_decisions
            WHERE external_id=? ORDER BY decided_at DESC,rowid DESC LIMIT 1
            """,
            (external_id,),
        ).fetchone()

    def record_decision(
        self,
        *,
        external_id: str,
        decision: str,
        reason: str,
        actor: str,
        packet_id: str | None = None,
        draft: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise SocialIntelligenceError(f"Unknown decision: {decision}")
        reason = reason.strip()
        if not reason or len(reason) > 500:
            raise SocialIntelligenceError(
                "A non-empty reason of at most 500 characters is required"
            )
        if not actor.strip():
            raise SocialIntelligenceError("An actor name is required")
        item = self._enriched_item(external_id)
        previous = self.latest_decision(external_id)
        if previous is not None and previous["decision"] == "accept":
            raise SocialIntelligenceError(
                f"Item {external_id} was already accepted "
                f"({previous['decision_id']}); the ledger is append-only and "
                "an accepted item cannot be re-decided"
            )
        decision_id = f"decision-{uuid.uuid4().hex[:16]}"
        decided_at = utc_now()
        result: dict[str, Any] = {
            "decision_id": decision_id,
            "decided_at": decided_at,
            "external_id": external_id,
            "decision": decision,
            "recommended_action": item["recommended_action"],
        }
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                INSERT INTO owner_review_decisions(
                    decision_id,decided_at,actor,external_id,packet_id,
                    decision,reason,recommended_action,enrichment_run_id,
                    content_sha256
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    decision_id,
                    decided_at,
                    actor.strip(),
                    external_id,
                    packet_id,
                    decision,
                    reason,
                    item["recommended_action"],
                    item["run_id"],
                    item["content_sha256"],
                ),
            )
            if decision == "accept":
                kind = QUEUE_KIND_BY_ACTION.get(item["recommended_action"])
                if kind is not None:
                    result["queue_id"] = self._enqueue(
                        item=item,
                        kind=kind,
                        decision_id=decision_id,
                        packet_id=packet_id,
                        actor=actor.strip(),
                        decided_at=decided_at,
                    )
                elif item["recommended_action"] == "reply_candidate":
                    result["draft_id"] = self._accept_reply(
                        item=item, draft=draft
                    )
                else:
                    raise SocialIntelligenceError(
                        "Accepting an item whose recommended_action is "
                        f"'{item['recommended_action']}' is not a governed "
                        "outcome; only experiment/investigate/reply classes "
                        "can be accepted"
                    )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return result

    def _enqueue(
        self,
        *,
        item: sqlite3.Row,
        kind: str,
        decision_id: str,
        packet_id: str | None,
        actor: str,
        decided_at: str,
    ) -> str:
        collision = self.connection.execute(
            """
            SELECT queue_id,external_id FROM social_work_queue
            WHERE external_id=? OR json_extract(provenance_json,
                '$.content_sha256')=?
            """,
            (item["external_id"], item["content_sha256"]),
        ).fetchone()
        if collision is not None:
            raise SocialIntelligenceError(
                f"Work-queue collision: {collision['queue_id']} already "
                f"covers item {collision['external_id']} or identical content"
            )
        queue_id = f"work-{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            """
            INSERT INTO social_work_queue(
                queue_id,created_at,kind,external_id,decision_id,title,
                source_url,claims_json,target_fronts_json,provenance_json,
                state
            ) VALUES (?,?,?,?,?,?,?,?,?,?,'queued')
            """,
            (
                queue_id,
                decided_at,
                kind,
                item["external_id"],
                decision_id,
                item["title"][:300],
                item["url"],
                item["claims_json"],
                item["target_fronts_json"],
                canonical_json(
                    {
                        "external_id": item["external_id"],
                        "source_url": item["url"],
                        "content_sha256": item["content_sha256"],
                        "enrichment_run_id": item["run_id"],
                        "analyzed_at": item["analyzed_at"],
                        "decision_id": decision_id,
                        "packet_id": packet_id,
                        "accepted_by": actor,
                        "accepted_at": decided_at,
                    }
                ),
            ),
        )
        return queue_id

    def _accept_reply(
        self, *, item: sqlite3.Row, draft: Mapping[str, str] | None
    ) -> str | None:
        """Accepted replies become DRAFTS only, and only from owner content.

        Without owner-authored draft content the acceptance is recorded and
        the draft stays pending; nothing is generated or published here."""
        if not draft:
            return None
        for key in ("title", "content", "submolt"):
            if not str(draft.get(key, "")).strip():
                raise SocialIntelligenceError(
                    f"Reply draft requires a non-empty {key}"
                )
        return self.store.create_draft(
            title=str(draft["title"]),
            content=str(draft["content"]),
            submolt=str(draft["submolt"]),
            source_ids=[item["external_id"]],
        )

    def decisions(self, external_id: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM owner_review_decisions"
        params: tuple[str, ...] = ()
        if external_id:
            query += " WHERE external_id=?"
            params = (external_id,)
        return self.connection.execute(
            query + " ORDER BY decided_at,rowid", params
        ).fetchall()

    def queue(self, state: str | None = None) -> list[sqlite3.Row]:
        query = "SELECT * FROM social_work_queue"
        params: tuple[str, ...] = ()
        if state:
            query += " WHERE state=?"
            params = (state,)
        return self.connection.execute(
            query + " ORDER BY created_at,queue_id", params
        ).fetchall()

    def status(self) -> dict[str, Any]:
        decisions = dict(
            self.connection.execute(
                "SELECT decision,COUNT(*) FROM owner_review_decisions "
                "GROUP BY decision"
            ).fetchall()
        )
        queue = dict(
            self.connection.execute(
                "SELECT kind,COUNT(*) FROM social_work_queue GROUP BY kind"
            ).fetchall()
        )
        return {
            "schema": LEDGER_SCHEMA,
            "decisions_by_kind": decisions,
            "work_queue_by_kind": queue,
            "decided_items": int(
                self.connection.execute(
                    "SELECT COUNT(DISTINCT external_id) "
                    "FROM owner_review_decisions"
                ).fetchone()[0]
            ),
        }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    sub = value.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide")
    decide.add_argument("--item", required=True)
    decide.add_argument("--decision", required=True, choices=DECISIONS)
    decide.add_argument("--reason", required=True)
    decide.add_argument("--actor", default="human-owner")
    decide.add_argument("--packet")
    decide.add_argument("--draft-title")
    decide.add_argument("--draft-content-file", type=Path)
    decide.add_argument("--draft-submolt")
    list_decisions = sub.add_parser("decisions")
    list_decisions.add_argument("--item")
    queue = sub.add_parser("queue")
    queue.add_argument("--state")
    sub.add_parser("status")
    return value


def main() -> int:
    args = parser().parse_args()
    config = SocialConfig.load(args.config)
    store = SocialOlap(config.database_path)
    ledger = ReviewLedger(store)
    try:
        if args.command == "decide":
            draft = None
            if args.draft_title or args.draft_content_file or args.draft_submolt:
                if not (
                    args.draft_title
                    and args.draft_content_file
                    and args.draft_submolt
                ):
                    raise SocialIntelligenceError(
                        "Reply drafts require --draft-title, "
                        "--draft-content-file and --draft-submolt together"
                    )
                draft = {
                    "title": args.draft_title,
                    "content": args.draft_content_file.read_text(
                        encoding="utf-8"
                    ),
                    "submolt": args.draft_submolt,
                }
            result: Any = ledger.record_decision(
                external_id=args.item,
                decision=args.decision,
                reason=args.reason,
                actor=args.actor,
                packet_id=args.packet,
                draft=draft,
            )
        elif args.command == "decisions":
            result = [dict(row) for row in ledger.decisions(args.item)]
        elif args.command == "queue":
            result = [dict(row) for row in ledger.queue(args.state)]
        else:
            result = ledger.status()
    except SocialIntelligenceError as exc:
        print(
            json.dumps({"status": "refused", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 1
    finally:
        store.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
