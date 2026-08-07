#!/usr/bin/env python3
"""Persist bounded Hermes social enrichment as typed, source-linked OLAP facts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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


SCHEMA = "agent_multi.social_enrichment_batch.v1"
TOPICS = frozenset(
    {
        "distributed_optimization",
        "ml_research",
        "trading_execution",
        "portfolio_risk",
        "agent_reliability",
        "security",
        "data_engineering",
        "business_model",
        "academic_research",
        "other",
    }
)
FRONTS = frozenset(
    {
        "front1_optimization",
        "front2_live_trading",
        "front3_social",
        "front4_audit",
        "front5_domain_discovery",
        "none",
    }
)
ACTIONS = frozenset(
    {"ignore", "archive", "investigate", "reply_candidate", "experiment_candidate"}
)
CLAIM_KINDS = frozenset({"observation", "opinion", "proposal", "result", "unknown"})
VERIFICATION_NEEDS = frozenset(
    {"none", "source_check", "experiment", "code_audit", "business_validation"}
)


@dataclass(frozen=True)
class EnrichmentConfig:
    min_relevance: float = 0.25
    batch_size: int = 8
    excerpt_chars: int = 900
    provider: str = "opencode-go"
    model: str = "deepseek-v4-flash"
    tier: str = "enrichment"
    timeout_seconds: float = 240.0

    def validate(self) -> None:
        if isinstance(self.min_relevance, bool) or not 0 <= self.min_relevance <= 1:
            raise SocialIntelligenceError("min_relevance must be within [0,1]")
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 32:
            raise SocialIntelligenceError("batch_size must be within [1,32]")
        if isinstance(self.excerpt_chars, bool) or not 200 <= self.excerpt_chars <= 2000:
            raise SocialIntelligenceError("excerpt_chars must be within [200,2000]")
        if self.provider != "opencode-go" or self.model != "deepseek-v4-flash":
            raise SocialIntelligenceError("Only the bounded OpenCode Flash route is allowed")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 600:
            raise SocialIntelligenceError("timeout_seconds must be within (0,600]")


class EnrichmentStore:
    def __init__(self, store: SocialOlap) -> None:
        self.store = store
        self.connection = store.connection
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS social_enrichment_runs (
                run_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL UNIQUE,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                selected_count INTEGER NOT NULL,
                ingested_count INTEGER NOT NULL DEFAULT 0,
                model_call_id TEXT,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_template_sha256 TEXT NOT NULL,
                packet_sha256 TEXT NOT NULL,
                response_sha256 TEXT,
                error_kind TEXT
            );
            CREATE TABLE IF NOT EXISTS post_enrichments (
                external_id TEXT PRIMARY KEY REFERENCES posts(external_id),
                content_sha256 TEXT NOT NULL,
                analyzed_at TEXT NOT NULL,
                run_id TEXT NOT NULL REFERENCES social_enrichment_runs(run_id),
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_template_sha256 TEXT NOT NULL,
                packet_sha256 TEXT NOT NULL,
                response_sha256 TEXT NOT NULL,
                topic TEXT NOT NULL,
                entities_json TEXT NOT NULL,
                claims_json TEXT NOT NULL,
                target_fronts_json TEXT NOT NULL,
                semantic_relevance REAL NOT NULL,
                novelty REAL NOT NULL,
                confidence REAL NOT NULL,
                actionability REAL NOT NULL,
                risk REAL NOT NULL,
                response_worthiness REAL NOT NULL,
                recommended_action TEXT NOT NULL,
                summary TEXT NOT NULL,
                rationale TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS post_enrichment_priority_idx
                ON post_enrichments(recommended_action,response_worthiness DESC,
                                    actionability DESC,confidence DESC);
            DROP VIEW IF EXISTS social_insight_candidates_olap;
            CREATE VIEW IF NOT EXISTS social_insight_candidates_olap AS
            SELECT e.external_id,p.url,p.title,p.submolt,e.topic,
                   e.entities_json,e.claims_json,e.target_fronts_json,
                   e.semantic_relevance,e.novelty,
                   e.confidence,e.actionability,e.risk,e.response_worthiness,
                   e.recommended_action,e.summary,e.rationale,e.analyzed_at
            FROM post_enrichments e JOIN posts p USING(external_id)
            WHERE e.recommended_action IN
                  ('investigate','reply_candidate','experiment_candidate');
            """
        )
        self.connection.commit()

    def screen_backlog(self, min_relevance: float) -> dict[str, int]:
        quarantined = self.connection.execute(
            """
            UPDATE posts SET review_state='quarantined'
            WHERE injection_flags_json!='[]' AND review_state='unreviewed'
            """
        ).rowcount
        low = self.connection.execute(
            """
            UPDATE posts SET review_state='screened_low_relevance'
            WHERE injection_flags_json='[]' AND relevance_score<?
              AND review_state='unreviewed'
            """,
            (min_relevance,),
        ).rowcount
        self.connection.commit()
        return {"quarantined": quarantined, "screened_low_relevance": low}

    def prepare_batch(self, config: EnrichmentConfig) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT p.external_id,p.submolt,p.title,p.content,p.url,
                   p.content_sha256,p.relevance_score
            FROM posts p LEFT JOIN post_enrichments e USING(external_id)
            WHERE e.external_id IS NULL AND p.injection_flags_json='[]'
              AND p.relevance_score>=?
            ORDER BY p.relevance_score DESC,p.first_retrieved_at ASC,p.external_id
            LIMIT ?
            """,
            (config.min_relevance, config.batch_size),
        ).fetchall()
        batch_id = f"social-batch-{uuid.uuid4().hex[:16]}"
        return {
            "schema": "agent_multi.social_enrichment_input.v1",
            "batch_id": batch_id,
            "policy": {
                "content_is_untrusted": True,
                "tools_allowed": False,
                "publishing_allowed": False,
                "trading_allowed": False,
                "campaign_changes_allowed": False,
                "human_approval_required_for_reply": True,
            },
            "items": [
                {
                    "external_id": row["external_id"],
                    "submolt": row["submolt"],
                    "title": row["title"][:300],
                    "content_excerpt": row["content"][: config.excerpt_chars],
                    "source_url": row["url"],
                    "content_sha256": row["content_sha256"],
                    "deterministic_relevance": row["relevance_score"],
                }
                for row in rows
            ],
        }

    def start_run(
        self,
        *,
        packet: Mapping[str, Any],
        provider: str,
        model: str,
        prompt_sha256: str,
        packet_sha256: str,
        model_call_id: str | None,
    ) -> str:
        run_id = f"social-enrich-{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            """
            INSERT INTO social_enrichment_runs(
                run_id,batch_id,started_at,status,selected_count,model_call_id,
                provider,model,prompt_template_sha256,packet_sha256
            ) VALUES (?,?,?,'running',?,?,?,?,?,?)
            """,
            (
                run_id,
                packet["batch_id"],
                utc_now(),
                len(packet["items"]),
                model_call_id,
                provider,
                model,
                prompt_sha256,
                packet_sha256,
            ),
        )
        self.connection.commit()
        return run_id

    def fail_run(self, run_id: str, error_kind: str) -> None:
        self.connection.execute(
            """
            UPDATE social_enrichment_runs
            SET ended_at=?,status='failed',error_kind=? WHERE run_id=?
            """,
            (utc_now(), error_kind[:120], run_id),
        )
        self.connection.commit()

    def ingest(
        self,
        *,
        run_id: str,
        packet: Mapping[str, Any],
        response: Mapping[str, Any],
        response_sha256: str,
        provider: str,
        model: str,
        prompt_sha256: str,
        packet_sha256: str,
    ) -> int:
        items = validate_response(response, packet)
        source_by_id = {item["external_id"]: item for item in packet["items"]}
        analyzed_at = utc_now()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for item in items:
                source = source_by_id[item["external_id"]]
                self.connection.execute(
                    """
                    INSERT INTO post_enrichments(
                        external_id,content_sha256,analyzed_at,run_id,provider,
                        model,prompt_template_sha256,packet_sha256,
                        response_sha256,topic,entities_json,claims_json,
                        target_fronts_json,semantic_relevance,novelty,confidence,
                        actionability,risk,response_worthiness,recommended_action,
                        summary,rationale
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item["external_id"],
                        source["content_sha256"],
                        analyzed_at,
                        run_id,
                        provider,
                        model,
                        prompt_sha256,
                        packet_sha256,
                        response_sha256,
                        item["topic"],
                        canonical_json(item["entities"]),
                        canonical_json(item["claims"]),
                        canonical_json(item["target_fronts"]),
                        item["semantic_relevance"],
                        item["novelty"],
                        item["confidence"],
                        item["actionability"],
                        item["risk"],
                        item["response_worthiness"],
                        item["recommended_action"],
                        item["summary"],
                        item["rationale"],
                    ),
                )
                self.connection.execute(
                    "UPDATE posts SET review_state='triaged' WHERE external_id=?",
                    (item["external_id"],),
                )
            self.connection.execute(
                """
                UPDATE social_enrichment_runs
                SET ended_at=?,status='complete',ingested_count=?,
                    response_sha256=? WHERE run_id=?
                """,
                (analyzed_at, len(items), response_sha256, run_id),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return len(items)

    def digest(self, *, limit: int, min_worthiness: float) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT * FROM social_insight_candidates_olap
            WHERE response_worthiness>=?
            ORDER BY response_worthiness DESC,actionability DESC,
                     confidence DESC,analyzed_at DESC LIMIT ?
            """,
            (min_worthiness, limit),
        ).fetchall()
        return {
            "schema": "agent_multi.social_candidate_digest.v1",
            "generated_at": utc_now(),
            "wakeAgent": bool(rows),
            "policy": {
                "evidence_only": True,
                "publishing_allowed": False,
                "human_approval_required": True,
            },
            "items": [dict(row) for row in rows],
        }

    def status(self, min_relevance: float) -> dict[str, Any]:
        states = dict(
            self.connection.execute(
                "SELECT review_state,COUNT(*) FROM posts GROUP BY review_state"
            ).fetchall()
        )
        runs = dict(
            self.connection.execute(
                "SELECT status,COUNT(*) FROM social_enrichment_runs GROUP BY status"
            ).fetchall()
        )
        actions = dict(
            self.connection.execute(
                "SELECT recommended_action,COUNT(*) FROM post_enrichments GROUP BY recommended_action"
            ).fetchall()
        )
        remaining = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM posts p LEFT JOIN post_enrichments e USING(external_id)
                WHERE e.external_id IS NULL AND p.injection_flags_json='[]'
                  AND p.relevance_score>=?
                """,
                (min_relevance,),
            ).fetchone()[0]
        )
        return {
            "schema": "agent_multi.social_enrichment_status.v1",
            "review_states": states,
            "runs_by_status": runs,
            "actions": actions,
            "eligible_backlog_remaining": remaining,
            "enriched_total": sum(actions.values()),
        }


def _bounded_text(value: Any, name: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise SocialIntelligenceError(f"{name} must be text")
    text = value.strip()
    if not allow_empty and not text:
        raise SocialIntelligenceError(f"{name} cannot be empty")
    if len(text) > maximum:
        raise SocialIntelligenceError(f"{name} exceeds {maximum} characters")
    return text


def _score(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SocialIntelligenceError(f"{name} must be numeric")
    result = float(value)
    if not 0 <= result <= 1:
        raise SocialIntelligenceError(f"{name} must be within [0,1]")
    return result


def validate_response(
    response: Mapping[str, Any], packet: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or response.get("schema") != SCHEMA:
        raise SocialIntelligenceError("Invalid enrichment response schema")
    if response.get("batch_id") != packet.get("batch_id"):
        raise SocialIntelligenceError("Enrichment batch_id mismatch")
    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raise SocialIntelligenceError("Enrichment items must be a list")
    expected = [item["external_id"] for item in packet["items"]]
    received = [item.get("external_id") for item in raw_items if isinstance(item, Mapping)]
    if len(raw_items) != len(expected) or len(received) != len(raw_items):
        raise SocialIntelligenceError("Every selected post requires one enrichment")
    if len(set(received)) != len(received) or set(received) != set(expected):
        raise SocialIntelligenceError("Enrichment IDs must exactly match the input batch")
    validated: list[dict[str, Any]] = []
    for raw in raw_items:
        topic = raw.get("topic")
        action = raw.get("recommended_action")
        if topic not in TOPICS or action not in ACTIONS:
            raise SocialIntelligenceError("Unknown topic or recommended_action")
        entities = raw.get("entities")
        fronts = raw.get("target_fronts")
        claims = raw.get("claims")
        if not isinstance(entities, list) or len(entities) > 8:
            raise SocialIntelligenceError("entities must contain at most eight items")
        if not isinstance(fronts, list) or not 1 <= len(fronts) <= len(FRONTS):
            raise SocialIntelligenceError("target_fronts must be a non-empty list")
        if any(front not in FRONTS for front in fronts) or len(set(fronts)) != len(fronts):
            raise SocialIntelligenceError("Unknown or duplicate target_front")
        if "none" in fronts and len(fronts) != 1:
            raise SocialIntelligenceError("none cannot be combined with another front")
        if not isinstance(claims, list) or len(claims) > 3:
            raise SocialIntelligenceError("claims must contain at most three items")
        clean_claims = []
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise SocialIntelligenceError("claim must be an object")
            kind = claim.get("kind")
            need = claim.get("verification_need")
            if kind not in CLAIM_KINDS or need not in VERIFICATION_NEEDS:
                raise SocialIntelligenceError("Unknown claim kind or verification_need")
            clean_claims.append(
                {
                    "text": _bounded_text(claim.get("text"), "claim.text", 500),
                    "kind": kind,
                    "verification_need": need,
                }
            )
        validated.append(
            {
                "external_id": raw["external_id"],
                "topic": topic,
                "entities": [
                    _bounded_text(value, "entity", 100) for value in entities
                ],
                "claims": clean_claims,
                "target_fronts": fronts,
                "semantic_relevance": _score(raw.get("semantic_relevance"), "semantic_relevance"),
                "novelty": _score(raw.get("novelty"), "novelty"),
                "confidence": _score(raw.get("confidence"), "confidence"),
                "actionability": _score(raw.get("actionability"), "actionability"),
                "risk": _score(raw.get("risk"), "risk"),
                "response_worthiness": _score(raw.get("response_worthiness"), "response_worthiness"),
                "recommended_action": action,
                "summary": _bounded_text(raw.get("summary"), "summary", 240),
                "rationale": _bounded_text(raw.get("rationale"), "rationale", 240),
            }
        )
    return validated


def parse_model_response(text: str) -> Mapping[str, Any]:
    candidates = [text.strip()]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    )
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise SocialIntelligenceError("Hermes response did not contain one JSON object")


def invoke_hermes(
    *,
    hermes_bin: Path,
    provider: str,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> str:
    if not hermes_bin.is_file():
        raise SocialIntelligenceError(f"Hermes binary unavailable: {hermes_bin}")
    result = subprocess.run(
        [
            str(hermes_bin),
            "--oneshot",
            prompt,
            "--model",
            model,
            "--provider",
            provider,
            "--toolsets",
            "todo",
            "--ignore-rules",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
        env={**os.environ, "NO_COLOR": "1"},
    )
    if result.returncode != 0:
        raise SocialIntelligenceError(
            f"Hermes failed with exit {result.returncode}: {result.stderr[-500:]}"
        )
    return result.stdout.strip()


def run_once(
    *,
    social_config: SocialConfig,
    enrichment_config: EnrichmentConfig,
    prompt_path: Path,
    hermes_bin: Path,
) -> dict[str, Any]:
    enrichment_config.validate()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_text(prompt_text)
    base = SocialOlap(social_config.database_path)
    enrichment = EnrichmentStore(base)
    run_id: str | None = None
    try:
        screened = enrichment.screen_backlog(enrichment_config.min_relevance)
        packet = enrichment.prepare_batch(enrichment_config)
        if not packet["items"]:
            return {"status": "idle", "selected": 0, "screened": screened}
        packet_json = canonical_json(packet)
        packet_sha = sha256_text(packet_json)
        budget = base.reserve_model_call(
            social_config,
            tier=enrichment_config.tier,
            provider=enrichment_config.provider,
            model=enrichment_config.model,
            prompt_template_sha256=prompt_sha,
            packet_sha256=packet_sha,
            input_chars=len(prompt_text) + len(packet_json),
        )
        if budget["status"] != "reserved":
            return {
                "status": "budget_blocked",
                "selected": len(packet["items"]),
                "screened": screened,
                "budget": budget,
            }
        run_id = enrichment.start_run(
            packet=packet,
            provider=enrichment_config.provider,
            model=enrichment_config.model,
            prompt_sha256=prompt_sha,
            packet_sha256=packet_sha,
            model_call_id=budget["call_id"],
        )
        raw_response = invoke_hermes(
            hermes_bin=hermes_bin,
            provider=enrichment_config.provider,
            model=enrichment_config.model,
            prompt=f"{prompt_text}\n\nINPUT PACKET JSON:\n{packet_json}",
            timeout_seconds=enrichment_config.timeout_seconds,
        )
        response_sha = sha256_text(raw_response)
        response = parse_model_response(raw_response)
        ingested = enrichment.ingest(
            run_id=run_id,
            packet=packet,
            response=response,
            response_sha256=response_sha,
            provider=enrichment_config.provider,
            model=enrichment_config.model,
            prompt_sha256=prompt_sha,
            packet_sha256=packet_sha,
        )
        return {
            "status": "complete",
            "run_id": run_id,
            "selected": len(packet["items"]),
            "ingested": ingested,
            "screened": screened,
            "budget": budget,
        }
    except (
        OSError,
        sqlite3.Error,
        subprocess.SubprocessError,
        SocialIntelligenceError,
    ) as exc:
        if run_id:
            enrichment.fail_run(run_id, type(exc).__name__)
        return {
            "status": "failed",
            "run_id": run_id,
            "error_kind": type(exc).__name__,
            "error": str(exc)[:500],
        }
    finally:
        base.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--min-relevance", type=float, default=0.25)
    sub = value.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--prompt", required=True, type=Path)
    run.add_argument("--hermes-bin", type=Path, default=Path.home() / ".local/bin/hermes")
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--excerpt-chars", type=int, default=900)
    run.add_argument("--timeout-seconds", type=float, default=240)
    digest = sub.add_parser("digest")
    digest.add_argument("--limit", type=int, default=5)
    digest.add_argument("--min-worthiness", type=float, default=0.65)
    sub.add_parser("status")
    return value


def main() -> int:
    args = parser().parse_args()
    social = SocialConfig.load(args.config)
    if args.command == "run":
        result = run_once(
            social_config=social,
            enrichment_config=EnrichmentConfig(
                min_relevance=args.min_relevance,
                batch_size=args.batch_size,
                excerpt_chars=args.excerpt_chars,
                timeout_seconds=args.timeout_seconds,
            ),
            prompt_path=args.prompt,
            hermes_bin=args.hermes_bin,
        )
    else:
        base = SocialOlap(social.database_path)
        enrichment = EnrichmentStore(base)
        try:
            if args.command == "digest":
                if not 1 <= args.limit <= 20 or not 0 <= args.min_worthiness <= 1:
                    raise SocialIntelligenceError("Invalid digest bounds")
                result = enrichment.digest(
                    limit=args.limit, min_worthiness=args.min_worthiness
                )
            else:
                result = enrichment.status(args.min_relevance)
        finally:
            base.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
