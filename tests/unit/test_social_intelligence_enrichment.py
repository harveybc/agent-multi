import json
from pathlib import Path

import pytest

from tools.social_intelligence import SocialConfig, SocialIntelligenceError, SocialOlap
from tools.social_intelligence_enrichment import (
    EnrichmentConfig,
    EnrichmentStore,
    parse_model_response,
    validate_response,
)


def _config(tmp_path: Path) -> SocialConfig:
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
            "daily_reserved_token_cap": 10000,
            "monthly_reserved_token_cap": 100000,
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


def _post(post_id: str, *, score: float, flagged: bool = False) -> dict:
    return {
        "external_id": post_id,
        "retrieval_source": "fixture",
        "submolt": "builds",
        "author": "builder",
        "title": "Optimization evidence",
        "content": "A bounded testable proposal",
        "url": f"https://www.moltbook.com/post/{post_id}",
        "published_at": "2026-08-07T00:00:00Z",
        "retrieved_at": "2026-08-07T00:01:00Z",
        "content_sha256": post_id.rjust(64, "a")[-64:],
        "raw_sha256": post_id.rjust(64, "b")[-64:],
        "injection_flags": ["prompt_injection"] if flagged else [],
        "score": score,
    }


def _response(packet: dict) -> dict:
    return {
        "schema": "agent_multi.social_enrichment_batch.v1",
        "batch_id": packet["batch_id"],
        "items": [
            {
                "external_id": item["external_id"],
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
            for item in packet["items"]
        ],
    }


def test_screening_and_batch_exclude_noise_and_injection(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        for value in (
            _post("high", score=0.8),
            _post("low", score=0.1),
            _post("flagged", score=0.9, flagged=True),
        ):
            base.ingest(value, value["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        assert store.screen_backlog(0.25) == {
            "quarantined": 1,
            "screened_low_relevance": 1,
        }
        packet = store.prepare_batch(EnrichmentConfig())
        assert [item["external_id"] for item in packet["items"]] == ["high"]
        states = dict(
            base.connection.execute(
                "SELECT external_id,review_state FROM posts"
            ).fetchall()
        )
        assert states["low"] == "screened_low_relevance"
        assert states["flagged"] == "quarantined"
    finally:
        base.close()


def test_response_requires_exact_ids_and_numeric_scores(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1", score=0.8)
        base.ingest(post, post["score"])
        base.connection.commit()
        packet = EnrichmentStore(base).prepare_batch(EnrichmentConfig())
        response = _response(packet)
        assert validate_response(response, packet)[0]["external_id"] == "p1"
        response["items"][0]["external_id"] = "foreign"
        with pytest.raises(SocialIntelligenceError, match="exactly match"):
            validate_response(response, packet)
        response = _response(packet)
        response["items"][0]["confidence"] = True
        with pytest.raises(SocialIntelligenceError, match="numeric"):
            validate_response(response, packet)
    finally:
        base.close()


def test_ingestion_is_atomic_and_updates_review_state(tmp_path):
    config = _config(tmp_path)
    base = SocialOlap(config.database_path)
    try:
        post = _post("p1", score=0.8)
        base.ingest(post, post["score"])
        base.connection.commit()
        store = EnrichmentStore(base)
        packet = store.prepare_batch(EnrichmentConfig())
        run_id = store.start_run(
            packet=packet,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="b" * 64,
            model_call_id="call-1",
        )
        assert store.ingest(
            run_id=run_id,
            packet=packet,
            response=_response(packet),
            response_sha256="c" * 64,
            provider="opencode-go",
            model="deepseek-v4-flash",
            prompt_sha256="a" * 64,
            packet_sha256="b" * 64,
        ) == 1
        assert base.connection.execute(
            "SELECT review_state FROM posts WHERE external_id='p1'"
        ).fetchone()[0] == "triaged"
        status = store.status(0.25)
        assert status["enriched_total"] == 1
        assert status["eligible_backlog_remaining"] == 0
        candidate = base.connection.execute(
            "SELECT entities_json,claims_json FROM social_insight_candidates_olap"
        ).fetchone()
        assert json.loads(candidate["entities_json"]) == ["DOIN"]
        assert json.loads(candidate["claims_json"])[0]["kind"] == "proposal"
    finally:
        base.close()


def test_parser_accepts_json_fence_but_not_prose_only():
    value = parse_model_response('```json\n{"schema":"x"}\n```')
    assert value["schema"] == "x"
    with pytest.raises(SocialIntelligenceError, match="JSON object"):
        parse_model_response("nothing structured")


def test_config_rejects_unapproved_model_and_bad_bounds():
    with pytest.raises(SocialIntelligenceError, match="OpenCode Flash"):
        EnrichmentConfig(model="deepseek-v4-pro").validate()
    with pytest.raises(SocialIntelligenceError, match="batch_size"):
        EnrichmentConfig(batch_size=0).validate()
