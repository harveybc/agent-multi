"""WP5 (order 2026-08-10 §8): idempotent enrichment retries, bounded review
packet materializer and append-only owner-review ledger. Fixtures only; no
network, no Hermes binary, no GPU."""

import json
from pathlib import Path

import pytest
import sqlite3

from tools.social_intelligence import (
    SocialConfig,
    SocialIntelligenceError,
    SocialOlap,
)
from tools.social_intelligence_enrichment import (
    EnrichmentConfig,
    EnrichmentStore,
    plan_failed_run_retries,
    retry_failed_runs,
)
from tools.social_review_ledger import ReviewLedger
from tools.social_review_packet import (
    materialize_packet,
    record_packet,
    render_markdown,
)


def _config(tmp_path: Path, *, daily_cap: int = 100000) -> SocialConfig:
    payload = {
        "schema": "agent_multi.social_intelligence_config.v1",
        "api_base_url": "https://www.moltbook.com/api/v1",
        "database_path": str(tmp_path / "social.sqlite"),
        "state_path": str(tmp_path / "state.json"),
        "secrets": {"api_key_env": "MOLTBOOK_API_KEY"},
        "collection": {
            "max_posts_per_run": 20,
            "sorts": ["new"],
            "submolts": [],
            "search_queries": [],
            "relevance_terms": ["optimization"],
        },
        "model_budget": {
            "daily_reserved_token_cap": daily_cap,
            "monthly_reserved_token_cap": 10 * daily_cap,
            "warning_ratio": 0.8,
            "tiers": {
                "enrichment": {
                    "max_input_tokens": 6000,
                    "reserved_output_tokens": 2500,
                }
            },
        },
        "publishing": {
            "enabled": False,
            "require_human_approval": True,
            "allowed_submolts": ["builds"],
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return SocialConfig.load(path)


def _post(post_id: str, *, score: float = 0.8) -> dict:
    return {
        "external_id": post_id,
        "retrieval_source": "fixture",
        "submolt": "builds",
        "author": "builder",
        "title": f"Optimization evidence {post_id}",
        "content": "A bounded testable proposal",
        "url": f"https://www.moltbook.com/post/{post_id}",
        "published_at": "2026-08-07T00:00:00Z",
        "retrieved_at": "2026-08-07T00:01:00Z",
        "content_sha256": post_id.rjust(64, "a")[-64:],
        "raw_sha256": post_id.rjust(64, "b")[-64:],
        "injection_flags": [],
        "score": score,
    }


def _enrichment_item(external_id: str, **overrides) -> dict:
    item = {
        "external_id": external_id,
        "topic": "distributed_optimization",
        "entities": ["DOIN"],
        "claims": [
            {
                "text": "The post proposes a bounded experiment.",
                "kind": "proposal",
                "verification_need": "experiment",
            }
        ],
        "target_fronts": ["front1_optimization"],
        "semantic_relevance": 0.9,
        "novelty": 0.7,
        "confidence": 0.8,
        "actionability": 0.9,
        "risk": 0.1,
        "response_worthiness": 0.75,
        "recommended_action": "experiment_candidate",
        "summary": "Bounded experiment proposal.",
        "rationale": "Specific and testable.",
    }
    item.update(overrides)
    return item


def _response(packet: dict, overrides_by_id: dict | None = None) -> dict:
    overrides_by_id = overrides_by_id or {}
    return {
        "schema": "agent_multi.social_enrichment_batch.v1",
        "batch_id": packet["batch_id"],
        "items": [
            _enrichment_item(
                item["external_id"],
                **overrides_by_id.get(item["external_id"], {}),
            )
            for item in packet["items"]
        ],
    }


def _echo_runner(overrides_by_id: dict | None = None):
    """Fake Hermes: parse the packet out of the prompt, answer validly."""

    def run(*, hermes_bin, provider, model, prompt, timeout_seconds):
        packet = json.loads(prompt.split("INPUT PACKET JSON:\n", 1)[1])
        return json.dumps(_response(packet, overrides_by_id))

    return run


def _failing_runner(*, hermes_bin, provider, model, prompt, timeout_seconds):
    raise SocialIntelligenceError("Hermes response did not contain one JSON object")


def _seed_failed_run(base: SocialOlap, store: EnrichmentStore) -> str:
    packet = store.prepare_batch(EnrichmentConfig())
    run_id = store.start_run(
        packet=packet,
        provider="opencode-go",
        model="deepseek-v4-flash",
        prompt_sha256="a" * 64,
        packet_sha256="b" * 64,
        model_call_id="call-original",
    )
    store.fail_run(run_id, "SocialIntelligenceError")
    return run_id


def _prompt(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text("Enrich the following packet.", encoding="utf-8")
    return path


def test_retry_preserves_run_id_error_class_attempts_and_budget(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1")
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        run_id = _seed_failed_run(base, store)
        reservations_before = base.connection.execute(
            "SELECT COUNT(*) FROM model_call_reservations"
        ).fetchone()[0]
        result = retry_failed_runs(
            social_config=config,
            enrichment_config=EnrichmentConfig(),
            prompt_path=_prompt(tmp_path),
            hermes_bin=tmp_path / "missing-hermes",
            runner=_echo_runner(),
        )
        assert result["status"] == "complete"
        assert result["outcomes"] == {"complete": 1}
        assert result["runs"][0]["run_id"] == run_id
        assert result["runs"][0]["original_error_kind"] == "SocialIntelligenceError"
        run = base.connection.execute(
            "SELECT * FROM social_enrichment_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        assert run["status"] == "complete"
        assert run["attempts"] == 2
        assert run["ingested_count"] == 1
        enriched = base.connection.execute(
            "SELECT run_id FROM post_enrichments WHERE external_id='p1'"
        ).fetchone()
        assert enriched["run_id"] == run_id
        attempts = store.run_attempts(run_id)
        assert [row["attempt"] for row in attempts] == [1, 2]
        assert attempts[0]["mode"] == "original"
        assert attempts[0]["status"] == "failed"
        assert attempts[0]["error_kind"] == "SocialIntelligenceError"
        assert attempts[1]["mode"] == "retry"
        assert attempts[1]["status"] == "complete"
        assert attempts[1]["model_call_id"]
        assert attempts[1]["reserved_total_tokens"] > 0
        reservations_after = base.connection.execute(
            "SELECT COUNT(*) FROM model_call_reservations"
        ).fetchone()[0]
        assert reservations_after == reservations_before + 1
        # Idempotency: nothing failed remains, second invocation is a no-op.
        again = retry_failed_runs(
            social_config=config,
            enrichment_config=EnrichmentConfig(),
            prompt_path=_prompt(tmp_path),
            hermes_bin=tmp_path / "missing-hermes",
            runner=_echo_runner(),
        )
        assert again["retried"] == 0
    finally:
        base.close()


def test_retry_resolves_superseded_without_model_call(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1")
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        failed_run_id = _seed_failed_run(base, store)
        # A later run absorbed the backlog, as happened on the live pipeline.
        packet = store.prepare_batch(EnrichmentConfig())
        later_run = store.start_run(
            packet=packet,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="c" * 64,
            model_call_id="call-later",
        )
        store.ingest(
            run_id=later_run,
            packet=packet,
            response=_response(packet),
            response_sha256="d" * 64,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="c" * 64,
        )
        reservations_before = base.connection.execute(
            "SELECT COUNT(*) FROM model_call_reservations"
        ).fetchone()[0]
        result = retry_failed_runs(
            social_config=config,
            enrichment_config=EnrichmentConfig(),
            prompt_path=_prompt(tmp_path),
            hermes_bin=tmp_path / "missing-hermes",
            runner=_failing_runner,  # must never be called
        )
        assert result["outcomes"] == {"superseded": 1}
        run = base.connection.execute(
            "SELECT status,attempts,error_kind FROM social_enrichment_runs "
            "WHERE run_id=?",
            (failed_run_id,),
        ).fetchone()
        assert run["status"] == "superseded"
        assert run["attempts"] == 2
        # Original error class stays journaled.
        assert run["error_kind"] == "SocialIntelligenceError"
        assert (
            base.connection.execute(
                "SELECT COUNT(*) FROM model_call_reservations"
            ).fetchone()[0]
            == reservations_before
        )
        assert store.run_attempts(failed_run_id)[1]["status"] == "superseded"
    finally:
        base.close()


def test_retry_budget_block_is_typed_and_stops_the_loop(tmp_path):
    config = _config(tmp_path, daily_cap=10)
    base = SocialOlap(config.database_path)
    try:
        for name in ("p1", "p2"):
            post = _post(name)
            base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        first = _seed_failed_run(base, store)
        second = _seed_failed_run(base, store)
        result = retry_failed_runs(
            social_config=config,
            enrichment_config=EnrichmentConfig(),
            prompt_path=_prompt(tmp_path),
            hermes_bin=tmp_path / "missing-hermes",
            runner=_echo_runner(),
        )
        assert result["status"] == "partial"
        assert result["outcomes"] == {"budget_blocked": 1}
        rows = {
            row["run_id"]: row
            for row in base.connection.execute(
                "SELECT run_id,status,attempts FROM social_enrichment_runs"
            )
        }
        assert rows[first]["status"] == "failed"
        assert rows[first]["attempts"] == 2
        # The loop stopped: the second failed run was not attempted.
        assert rows[second]["status"] == "failed"
        assert rows[second]["attempts"] == 1
        attempt = store.run_attempts(first)[1]
        assert attempt["status"] == "budget_blocked"
        assert attempt["error_kind"] == "daily_reserved_token_cap_exceeded"
    finally:
        base.close()


def test_retry_failure_appends_attempt_and_preserves_original_class(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1")
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        run_id = _seed_failed_run(base, store)
        result = retry_failed_runs(
            social_config=config,
            enrichment_config=EnrichmentConfig(),
            prompt_path=_prompt(tmp_path),
            hermes_bin=tmp_path / "missing-hermes",
            runner=_failing_runner,
        )
        assert result["status"] == "partial"
        assert result["runs"][0]["outcome"] == "failed"
        run = base.connection.execute(
            "SELECT status,attempts,error_kind FROM social_enrichment_runs "
            "WHERE run_id=?",
            (run_id,),
        ).fetchone()
        assert run["status"] == "failed"
        assert run["attempts"] == 2
        assert run["error_kind"] == "SocialIntelligenceError"
        attempts = store.run_attempts(run_id)
        assert attempts[1]["status"] == "failed"
        assert attempts[1]["error_kind"] == "SocialIntelligenceError"
        assert "JSON object" in attempts[1]["error_detail"]
    finally:
        base.close()


def test_dry_run_plan_reports_slices_and_writes_nothing(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1")
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        first = _seed_failed_run(base, store)
        second = _seed_failed_run(base, store)
        plan = plan_failed_run_retries(
            social_config=config,
            enrichment_config=EnrichmentConfig(batch_size=1),
            prompt_path=_prompt(tmp_path),
        )
        assert plan["mode"] == "dry_run"
        assert plan["failed_runs_planned"] == 2
        assert plan["planned_outcomes"] == {
            "retry_model_call": 1,
            "superseded": 1,
            "would_block": 0,
        }
        by_run = {item["run_id"]: item for item in plan["runs"]}
        assert by_run[first]["planned_outcome"] == "retry_model_call"
        assert by_run[first]["planned_items"][0]["external_id"] == "p1"
        assert by_run[first]["reserved_total_tokens"] > 0
        assert by_run[first]["original_error_kind"] == "SocialIntelligenceError"
        assert by_run[second]["planned_outcome"] == "superseded"
        assert plan["policy"]["execution"] == "pending_owner_scheduler_window"
        # Nothing was mutated by planning.
        assert (
            base.connection.execute(
                "SELECT COUNT(*) FROM social_enrichment_run_attempts"
            ).fetchone()[0]
            == 0
        )
        assert (
            base.connection.execute(
                "SELECT COUNT(*) FROM model_call_reservations"
            ).fetchone()[0]
            == 0
        )
        statuses = [
            row["status"]
            for row in base.connection.execute(
                "SELECT status FROM social_enrichment_runs"
            )
        ]
        assert statuses == ["failed", "failed"]
    finally:
        base.close()


def _seed_enrichments(base: SocialOlap, store: EnrichmentStore) -> None:
    overrides = {
        "exp-hi": {"actionability": 0.95},
        "exp-lo": {"actionability": 0.4},
        "reply-1": {
            "recommended_action": "reply_candidate",
            "response_worthiness": 0.9,
        },
        "inv-hi": {
            "recommended_action": "investigate",
            "actionability": 0.9,
            "confidence": 0.9,
            "novelty": 0.9,
        },
        "inv-mid": {
            "recommended_action": "investigate",
            "actionability": 0.6,
            "confidence": 0.6,
            "novelty": 0.6,
        },
        "inv-lo": {
            "recommended_action": "investigate",
            "actionability": 0.2,
            "confidence": 0.2,
            "novelty": 0.2,
        },
        "noise": {"recommended_action": "ignore"},
    }
    for name in overrides:
        post = _post(name)
        base.ingest(post, post["score"])
    base.connection.commit()
    packet = store.prepare_batch(EnrichmentConfig(batch_size=len(overrides)))
    run_id = store.start_run(
        packet=packet,
        provider="opencode-go",
        model="deepseek-v4-flash",
        prompt_sha256="a" * 64,
        packet_sha256="b" * 64,
        model_call_id="call-1",
    )
    store.ingest(
        run_id=run_id,
        packet=packet,
        response=_response(packet, overrides),
        response_sha256="c" * 64,
        provider="opencode-go",
        model="deepseek-v4-flash",
        prompt_sha256="a" * 64,
        packet_sha256="b" * 64,
    )


def test_review_packet_is_bounded_ranked_and_flagged(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        _seed_enrichments(base, EnrichmentStore(base))
        packet = materialize_packet(base, top_n=2)
        assert packet["included_by_class"] == {
            "experiment_candidate": 2,
            "reply_candidate": 1,
            "investigate": 2,
        }
        assert packet["totals_by_class"]["investigate"] == 3
        experiments = packet["classes"]["experiment_candidate"]
        assert experiments[0]["external_id"] == "exp-hi"
        investigations = packet["classes"]["investigate"]
        assert [item["external_id"] for item in investigations] == [
            "inv-hi",
            "inv-mid",
        ]
        assert investigations[0]["value_score"] == pytest.approx(0.9)
        item = experiments[0]
        assert item["untrusted_content"] is True
        assert item["source_url"].startswith("https://www.moltbook.com/post/")
        assert item["claims"][0]["kind"] == "proposal"
        assert "work queue" in item["proposed_next_action"]
        assert "DRAFT" in packet["classes"]["reply_candidate"][0][
            "proposed_next_action"
        ]
        assert packet["policy"]["auto_execution_allowed"] is False
        assert packet["policy"]["publishing_allowed"] is False
        # The ignore-class row must never surface in a review packet.
        surfaced = {
            entry["external_id"]
            for items in packet["classes"].values()
            for entry in items
        }
        assert "noise" not in surfaced
        markdown = render_markdown(packet)
        assert "exp-hi" in markdown and "Untrusted content: True" in markdown
        assert "drafts only" in markdown
        record_packet(base, packet)
        stored = base.connection.execute(
            "SELECT * FROM social_review_packets"
        ).fetchone()
        assert stored["packet_id"] == packet["packet_id"]
        assert json.loads(stored["item_ids_json"])["investigate"] == [
            "inv-hi",
            "inv-mid",
        ]
        with pytest.raises(SocialIntelligenceError, match="top_n"):
            materialize_packet(base, top_n=0)
    finally:
        base.close()


def test_ledger_accept_experiment_enqueues_once_with_provenance(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        _seed_enrichments(base, EnrichmentStore(base))
        ledger = ReviewLedger(base)
        result = ledger.record_decision(
            external_id="exp-hi",
            decision="accept",
            reason="Direct relevance to Front 1 factorial design",
            actor="human-owner",
            packet_id="review-packet-test",
        )
        assert result["queue_id"].startswith("work-")
        queued = ledger.queue()[0]
        assert queued["kind"] == "experiment"
        assert queued["state"] == "queued"
        provenance = json.loads(queued["provenance_json"])
        assert provenance["decision_id"] == result["decision_id"]
        assert provenance["packet_id"] == "review-packet-test"
        assert provenance["content_sha256"]
        assert provenance["enrichment_run_id"].startswith("social-enrich-")
        # Collision checks: same item cannot be accepted twice.
        with pytest.raises(SocialIntelligenceError, match="already accepted"):
            ledger.record_decision(
                external_id="exp-hi",
                decision="accept",
                reason="duplicate",
                actor="human-owner",
            )
        # Accepted investigate rows enter the queue as investigations.
        investigation = ledger.record_decision(
            external_id="inv-hi",
            decision="accept",
            reason="Verify the claimed result against our own data",
            actor="human-owner",
        )
        kinds = {row["kind"] for row in ledger.queue()}
        assert kinds == {"experiment", "investigation"}
        assert investigation["queue_id"] != result["queue_id"]
    finally:
        base.close()


def test_ledger_accepted_reply_becomes_draft_only(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        _seed_enrichments(base, EnrichmentStore(base))
        ledger = ReviewLedger(base)
        # Acceptance without owner-authored content records the decision but
        # creates nothing.
        result = ledger.record_decision(
            external_id="reply-1",
            decision="accept",
            reason="Worth an owner-written reply",
            actor="human-owner",
        )
        assert result.get("draft_id") is None
        assert ledger.queue() == []
        # A second accept is refused even for replies (append-only).
        with pytest.raises(SocialIntelligenceError, match="already accepted"):
            ledger.record_decision(
                external_id="reply-1",
                decision="accept",
                reason="again",
                actor="human-owner",
                draft={
                    "title": "Reply",
                    "content": "Owner-authored reply text",
                    "submolt": "builds",
                },
            )
        # Fresh item accepted WITH owner content becomes a draft, state
        # 'draft', never published by this path.
        _ = ledger.record_decision(
            external_id="exp-lo",
            decision="reject",
            reason="Not actionable enough",
            actor="human-owner",
        )
        drafts_before = base.connection.execute(
            "SELECT COUNT(*) FROM drafts"
        ).fetchone()[0]
        assert drafts_before == 0
        # Use a reply-class item seeded separately.
        post = _post("reply-2")
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        packet = store.prepare_batch(EnrichmentConfig())
        run_id = store.start_run(
            packet=packet,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="e" * 64,
            model_call_id="call-2",
        )
        store.ingest(
            run_id=run_id,
            packet=packet,
            response=_response(
                packet,
                {"reply-2": {"recommended_action": "reply_candidate"}},
            ),
            response_sha256="f" * 64,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="e" * 64,
        )
        accepted = ledger.record_decision(
            external_id="reply-2",
            decision="accept",
            reason="Owner will reply with our evidence",
            actor="human-owner",
            draft={
                "title": "Our measured results",
                "content": "Owner-authored reply text with evidence.",
                "submolt": "builds",
            },
        )
        draft = base.connection.execute(
            "SELECT state,submolt,source_post_ids_json FROM drafts "
            "WHERE draft_id=?",
            (accepted["draft_id"],),
        ).fetchone()
        assert draft["state"] == "draft"
        assert json.loads(draft["source_post_ids_json"]) == ["reply-2"]
        published = base.connection.execute(
            "SELECT COUNT(*) FROM drafts WHERE state='published'"
        ).fetchone()[0]
        assert published == 0
    finally:
        base.close()


def test_ledger_is_append_only_and_fails_closed(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        _seed_enrichments(base, EnrichmentStore(base))
        ledger = ReviewLedger(base)
        with pytest.raises(SocialIntelligenceError, match="not in the enrichment"):
            ledger.record_decision(
                external_id="ghost",
                decision="accept",
                reason="x",
                actor="human-owner",
            )
        with pytest.raises(SocialIntelligenceError, match="reason"):
            ledger.record_decision(
                external_id="exp-hi",
                decision="reject",
                reason="   ",
                actor="human-owner",
            )
        with pytest.raises(SocialIntelligenceError, match="Unknown decision"):
            ledger.record_decision(
                external_id="exp-hi",
                decision="approve",
                reason="x",
                actor="human-owner",
            )
        # Accepting an ignore/archive-class item is not a governed outcome.
        with pytest.raises(SocialIntelligenceError, match="not a governed"):
            ledger.record_decision(
                external_id="noise",
                decision="accept",
                reason="x",
                actor="human-owner",
            )
        deferred = ledger.record_decision(
            external_id="exp-hi",
            decision="defer",
            reason="Wait for the factorial to finish",
            actor="human-owner",
        )
        # Defer can be superseded by a later append; history is preserved.
        ledger.record_decision(
            external_id="exp-hi",
            decision="reject",
            reason="Superseded by internal experiment",
            actor="human-owner",
        )
        history = ledger.decisions("exp-hi")
        assert [row["decision"] for row in history] == ["defer", "reject"]
        assert history[0]["decision_id"] == deferred["decision_id"]
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            base.connection.execute(
                "UPDATE owner_review_decisions SET reason='edited'"
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            base.connection.execute("DELETE FROM owner_review_decisions")
        status = ledger.status()
        assert status["decisions_by_kind"] == {"defer": 1, "reject": 1}
        assert status["decided_items"] == 1
    finally:
        base.close()
