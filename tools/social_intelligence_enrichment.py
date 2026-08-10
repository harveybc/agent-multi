#!/usr/bin/env python3
"""Persist bounded Hermes social enrichment as typed, source-linked OLAP facts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
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
            CREATE TABLE IF NOT EXISTS social_enrichment_run_attempts (
                run_id TEXT NOT NULL REFERENCES social_enrichment_runs(run_id),
                attempt INTEGER NOT NULL,
                mode TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                error_kind TEXT,
                error_detail TEXT,
                batch_id TEXT NOT NULL,
                packet_sha256 TEXT NOT NULL,
                response_sha256 TEXT,
                model_call_id TEXT,
                reserved_total_tokens INTEGER,
                PRIMARY KEY(run_id,attempt)
            );
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
        columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(social_enrichment_runs)"
            )
        }
        if "attempts" not in columns:
            self.connection.execute(
                "ALTER TABLE social_enrichment_runs "
                "ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1"
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
        rows = self.eligible_backlog_slice(
            min_relevance=config.min_relevance, limit=config.batch_size
        )
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

    def failed_runs(
        self, run_ids: Sequence[str] | None = None
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM social_enrichment_runs WHERE status='failed'"
        params: tuple[str, ...] = ()
        if run_ids:
            query += f" AND run_id IN ({','.join('?' for _ in run_ids)})"
            params = tuple(run_ids)
        return self.connection.execute(
            query + " ORDER BY started_at,run_id", params
        ).fetchall()

    def record_original_attempt(self, run: Mapping[str, Any]) -> None:
        """Backfill attempt #1 of a run into the append-only attempt journal."""
        self.connection.execute(
            """
            INSERT OR IGNORE INTO social_enrichment_run_attempts(
                run_id,attempt,mode,started_at,ended_at,status,error_kind,
                error_detail,batch_id,packet_sha256,response_sha256,
                model_call_id,reserved_total_tokens
            ) VALUES (?,1,'original',?,?,?,?,NULL,?,?,?,?,NULL)
            """,
            (
                run["run_id"],
                run["started_at"],
                run["ended_at"],
                run["status"],
                run["error_kind"],
                run["batch_id"],
                run["packet_sha256"],
                run["response_sha256"],
                run["model_call_id"],
            ),
        )

    def run_attempts(self, run_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT * FROM social_enrichment_run_attempts
            WHERE run_id=? ORDER BY attempt
            """,
            (run_id,),
        ).fetchall()

    def eligible_backlog_slice(
        self, *, min_relevance: float, limit: int, offset: int = 0
    ) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT p.external_id,p.submolt,p.title,p.content,p.url,
                   p.content_sha256,p.relevance_score
            FROM posts p LEFT JOIN post_enrichments e USING(external_id)
            WHERE e.external_id IS NULL AND p.injection_flags_json='[]'
              AND p.relevance_score>=?
            ORDER BY p.relevance_score DESC,p.first_retrieved_at ASC,p.external_id
            LIMIT ? OFFSET ?
            """,
            (min_relevance, limit, offset),
        ).fetchall()

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


def _bounded_generated_text(value: Any, name: str, maximum: int) -> str:
    """Normalize non-authoritative model prose without weakening its schema."""
    if not isinstance(value, str):
        raise SocialIntelligenceError(f"{name} must be text")
    text = value.strip()
    if not text:
        raise SocialIntelligenceError(f"{name} cannot be empty")
    return text if len(text) <= maximum else text[:maximum].rstrip()


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
                "summary": _bounded_generated_text(
                    raw.get("summary"), "summary", 240
                ),
                "rationale": _bounded_generated_text(
                    raw.get("rationale"), "rationale", 240
                ),
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


def plan_failed_run_retries(
    *,
    social_config: SocialConfig,
    enrichment_config: EnrichmentConfig,
    prompt_path: Path,
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Dry-run: report exactly what a retry would do, with no model call and
    no data mutation. Each failed run is assigned the next unenriched slice of
    the current eligible backlog; budget effects are projected cumulatively."""
    enrichment_config.validate()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    tier = social_config.model_tiers.get(enrichment_config.tier) or {}
    reserved_output = int(tier.get("reserved_output_tokens", 0))
    max_input = int(tier.get("max_input_tokens", 0))
    if reserved_output <= 0 or max_input <= 0:
        raise SocialIntelligenceError(
            f"Model tier {enrichment_config.tier} has an invalid token budget"
        )
    base = SocialOlap(social_config.database_path)
    enrichment = EnrichmentStore(base)
    try:
        now = datetime.now(timezone.utc)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = day_start.replace(day=1)
        daily_before, monthly_before = (
            int(
                base.connection.execute(
                    """
                    SELECT COALESCE(SUM(reserved_total_tokens),0)
                    FROM model_call_reservations
                    WHERE status='reserved' AND reserved_at>=?
                    """,
                    (start.isoformat(),),
                ).fetchone()[0]
            )
            for start in (day_start, month_start)
        )
        backlog_total = int(
            base.connection.execute(
                """
                SELECT COUNT(*) FROM posts p
                LEFT JOIN post_enrichments e USING(external_id)
                WHERE e.external_id IS NULL AND p.injection_flags_json='[]'
                  AND p.relevance_score>=?
                """,
                (enrichment_config.min_relevance,),
            ).fetchone()[0]
        )
        daily_after, monthly_after = daily_before, monthly_before
        planned_runs: list[dict[str, Any]] = []
        counts = {"retry_model_call": 0, "superseded": 0, "would_block": 0}
        for offset_index, run in enumerate(enrichment.failed_runs(run_ids)):
            rows = enrichment.eligible_backlog_slice(
                min_relevance=enrichment_config.min_relevance,
                limit=enrichment_config.batch_size,
                offset=offset_index * enrichment_config.batch_size,
            )
            plan: dict[str, Any] = {
                "run_id": run["run_id"],
                "original_batch_id": run["batch_id"],
                "original_started_at": run["started_at"],
                "original_error_kind": run["error_kind"],
                "original_selected_count": run["selected_count"],
                "attempts_so_far": run["attempts"],
                "planned_items": [
                    {
                        "external_id": row["external_id"],
                        "url": row["url"],
                        "title": row["title"][:120],
                        "relevance_score": row["relevance_score"],
                    }
                    for row in rows
                ],
            }
            if not rows:
                plan["planned_outcome"] = "superseded"
                counts["superseded"] += 1
                planned_runs.append(plan)
                continue
            packet_probe = {
                "schema": "agent_multi.social_enrichment_input.v1",
                "batch_id": "social-batch-dryrun0000000000",
                "items": [
                    {
                        "external_id": row["external_id"],
                        "submolt": row["submolt"],
                        "title": row["title"][:300],
                        "content_excerpt": row["content"][
                            : enrichment_config.excerpt_chars
                        ],
                        "source_url": row["url"],
                        "content_sha256": row["content_sha256"],
                        "deterministic_relevance": row["relevance_score"],
                    }
                    for row in rows
                ],
            }
            input_chars = len(prompt_text) + len(canonical_json(packet_probe))
            estimated_input = max(1, math.ceil(input_chars / 4))
            reserved_total = estimated_input + reserved_output
            daily_after += reserved_total
            monthly_after += reserved_total
            block_reason = None
            if estimated_input > max_input:
                block_reason = "tier_input_cap_exceeded"
            elif daily_after > social_config.daily_reserved_token_cap:
                block_reason = "daily_reserved_token_cap_exceeded"
            elif monthly_after > social_config.monthly_reserved_token_cap:
                block_reason = "monthly_reserved_token_cap_exceeded"
            plan.update(
                {
                    "planned_outcome": (
                        "would_block" if block_reason else "retry_model_call"
                    ),
                    "estimated_input_tokens": estimated_input,
                    "reserved_total_tokens": reserved_total,
                    "projected_daily_reserved_after": daily_after,
                    "projected_monthly_reserved_after": monthly_after,
                    "projected_block_reason": block_reason,
                }
            )
            counts["would_block" if block_reason else "retry_model_call"] += 1
            planned_runs.append(plan)
        return {
            "schema": "agent_multi.social_enrichment_retry_plan.v1",
            "mode": "dry_run",
            "generated_at": utc_now(),
            "database_path": str(social_config.database_path),
            "min_relevance": enrichment_config.min_relevance,
            "batch_size": enrichment_config.batch_size,
            "eligible_backlog_total": backlog_total,
            "budget_before": {
                "daily_reserved_tokens": daily_before,
                "daily_reserved_token_cap": social_config.daily_reserved_token_cap,
                "monthly_reserved_tokens": monthly_before,
                "monthly_reserved_token_cap": social_config.monthly_reserved_token_cap,
            },
            "failed_runs_planned": len(planned_runs),
            "planned_outcomes": counts,
            "runs": planned_runs,
            "policy": {
                "model_calls_performed": 0,
                "state_mutated": False,
                "execution": "pending_owner_scheduler_window",
            },
        }
    finally:
        base.close()


def retry_failed_runs(
    *,
    social_config: SocialConfig,
    enrichment_config: EnrichmentConfig,
    prompt_path: Path,
    hermes_bin: Path,
    run_ids: Sequence[str] | None = None,
    runner: Any = None,
) -> dict[str, Any]:
    """Idempotently retry failed enrichment runs under their ORIGINAL run IDs.

    Contract:
    - the original run row (run_id, batch_id, started_at) is never rewritten;
      each retry is one append-only row in social_enrichment_run_attempts and
      one increment of social_enrichment_runs.attempts;
    - the original error class stays journaled as attempt #1 (mode=original);
    - a retry whose backlog slice is empty resolves the run as 'superseded'
      (its work was absorbed by later complete runs) with no model call and
      no token reservation;
    - every model-calling retry reserves tokens through the existing
      reserve_model_call contract before invoking Hermes; a blocked budget
      stops the loop with a typed 'budget_blocked' attempt;
    - post-level idempotency comes from the post_enrichments primary key,
      so a duplicate retry can never double-ingest an enrichment.
    """
    enrichment_config.validate()
    call_model = runner or invoke_hermes
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_sha = sha256_text(prompt_text)
    base = SocialOlap(social_config.database_path)
    enrichment = EnrichmentStore(base)
    results: list[dict[str, Any]] = []
    try:
        for run in enrichment.failed_runs(run_ids):
            run_id = run["run_id"]
            attempt = int(run["attempts"]) + 1
            enrichment.record_original_attempt(run)
            packet = enrichment.prepare_batch(enrichment_config)
            packet_json = canonical_json(packet)
            packet_sha = sha256_text(packet_json)
            started_at = utc_now()
            if not packet["items"]:
                ended = utc_now()
                enrichment.connection.execute(
                    """
                    INSERT INTO social_enrichment_run_attempts(
                        run_id,attempt,mode,started_at,ended_at,status,
                        batch_id,packet_sha256
                    ) VALUES (?,?,'retry',?,?,'superseded',?,?)
                    """,
                    (run_id, attempt, started_at, ended, packet["batch_id"], packet_sha),
                )
                enrichment.connection.execute(
                    """
                    UPDATE social_enrichment_runs
                    SET status='superseded',ended_at=?,attempts=? WHERE run_id=?
                    """,
                    (ended, attempt, run_id),
                )
                enrichment.connection.commit()
                results.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "outcome": "superseded",
                        "original_error_kind": run["error_kind"],
                    }
                )
                continue
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
                enrichment.connection.execute(
                    """
                    INSERT INTO social_enrichment_run_attempts(
                        run_id,attempt,mode,started_at,ended_at,status,
                        error_kind,batch_id,packet_sha256,model_call_id,
                        reserved_total_tokens
                    ) VALUES (?,?,'retry',?,?,'budget_blocked',?,?,?,?,?)
                    """,
                    (
                        run_id,
                        attempt,
                        started_at,
                        utc_now(),
                        budget["block_reason"],
                        packet["batch_id"],
                        packet_sha,
                        budget["call_id"],
                        budget["reserved_total_tokens"],
                    ),
                )
                enrichment.connection.execute(
                    "UPDATE social_enrichment_runs SET attempts=? WHERE run_id=?",
                    (attempt, run_id),
                )
                enrichment.connection.commit()
                results.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "outcome": "budget_blocked",
                        "block_reason": budget["block_reason"],
                        "original_error_kind": run["error_kind"],
                    }
                )
                break
            enrichment.connection.execute(
                """
                INSERT INTO social_enrichment_run_attempts(
                    run_id,attempt,mode,started_at,status,batch_id,
                    packet_sha256,model_call_id,reserved_total_tokens
                ) VALUES (?,?,'retry',?,'running',?,?,?,?)
                """,
                (
                    run_id,
                    attempt,
                    started_at,
                    packet["batch_id"],
                    packet_sha,
                    budget["call_id"],
                    budget["reserved_total_tokens"],
                ),
            )
            enrichment.connection.commit()
            try:
                raw_response = call_model(
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
                enrichment.connection.execute(
                    """
                    UPDATE social_enrichment_run_attempts
                    SET ended_at=?,status='complete',response_sha256=?
                    WHERE run_id=? AND attempt=?
                    """,
                    (utc_now(), response_sha, run_id, attempt),
                )
                enrichment.connection.execute(
                    "UPDATE social_enrichment_runs SET attempts=? WHERE run_id=?",
                    (attempt, run_id),
                )
                enrichment.connection.commit()
                results.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "outcome": "complete",
                        "ingested": ingested,
                        "original_error_kind": run["error_kind"],
                    }
                )
            except (
                OSError,
                sqlite3.Error,
                subprocess.SubprocessError,
                SocialIntelligenceError,
            ) as exc:
                enrichment.connection.rollback()
                enrichment.connection.execute(
                    """
                    UPDATE social_enrichment_run_attempts
                    SET ended_at=?,status='failed',error_kind=?,error_detail=?
                    WHERE run_id=? AND attempt=?
                    """,
                    (
                        utc_now(),
                        type(exc).__name__,
                        str(exc)[:500],
                        run_id,
                        attempt,
                    ),
                )
                enrichment.connection.execute(
                    "UPDATE social_enrichment_runs SET attempts=? WHERE run_id=?",
                    (attempt, run_id),
                )
                enrichment.connection.commit()
                results.append(
                    {
                        "run_id": run_id,
                        "attempt": attempt,
                        "outcome": "failed",
                        "error_kind": type(exc).__name__,
                        "error": str(exc)[:500],
                        "original_error_kind": run["error_kind"],
                    }
                )
        outcomes: dict[str, int] = {}
        for item in results:
            outcomes[item["outcome"]] = outcomes.get(item["outcome"], 0) + 1
        return {
            "schema": "agent_multi.social_enrichment_retry_result.v1",
            "mode": "execute",
            "generated_at": utc_now(),
            "status": "complete" if all(
                item["outcome"] in {"complete", "superseded"} for item in results
            ) else "partial",
            "retried": len(results),
            "outcomes": outcomes,
            "runs": results,
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
    retry = sub.add_parser("retry-failed")
    retry.add_argument("--prompt", required=True, type=Path)
    retry.add_argument("--hermes-bin", type=Path, default=Path.home() / ".local/bin/hermes")
    retry.add_argument("--batch-size", type=int, default=8)
    retry.add_argument("--excerpt-chars", type=int, default=900)
    retry.add_argument("--timeout-seconds", type=float, default=240)
    retry.add_argument("--run-id", action="append", default=[])
    retry.add_argument(
        "--execute",
        action="store_true",
        help="Actually reserve budget and call Hermes; default is a dry-run plan",
    )
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
    elif args.command == "retry-failed":
        retry_config = EnrichmentConfig(
            min_relevance=args.min_relevance,
            batch_size=args.batch_size,
            excerpt_chars=args.excerpt_chars,
            timeout_seconds=args.timeout_seconds,
        )
        if args.execute:
            result = retry_failed_runs(
                social_config=social,
                enrichment_config=retry_config,
                prompt_path=args.prompt,
                hermes_bin=args.hermes_bin,
                run_ids=args.run_id or None,
            )
        else:
            result = plan_failed_run_retries(
                social_config=social,
                enrichment_config=retry_config,
                prompt_path=args.prompt,
                run_ids=args.run_id or None,
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
    return 0 if result.get("status") not in {"failed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
