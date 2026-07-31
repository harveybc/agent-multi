#!/usr/bin/env python3
"""Bounded Moltbook collection, social OLAP, review packets and publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "agent_multi.social_intelligence_olap.v1"
CONFIG_SCHEMA = "agent_multi.social_intelligence_config.v1"
MOLTBOOK_BASE_URL = "https://www.moltbook.com/api/v1"
USER_AGENT = "agent-multi-social-intelligence/1.0"

INJECTION_PATTERNS = (
    re.compile(r"\bignore (?:all |any )?(?:previous|prior|system) instructions?\b", re.I),
    re.compile(r"\b(system prompt|developer message|hidden instructions?)\b", re.I),
    re.compile(r"\b(send|upload|reveal|print|exfiltrate)\b.{0,48}\b(api key|token|secret|credential)", re.I),
    re.compile(r"\b(run|execute|open)\b.{0,32}\b(shell|terminal|command|tool)\b", re.I),
    re.compile(r"\b(disable|bypass|override)\b.{0,32}\b(safety|guard|policy|approval)\b", re.I),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def load_secret_env_value(name: str, env_file: Path) -> str:
    """Read one allowlisted value without evaluating the env file as shell code."""
    current = os.environ.get(name, "").strip()
    if current:
        return current
    if not env_file.is_file():
        return ""
    if env_file.stat().st_mode & 0o077:
        raise SocialIntelligenceError(
            f"Credential file permissions are too broad: {env_file}; require mode 0600"
        )
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip("\"'").strip()
    return ""


class SocialIntelligenceError(RuntimeError):
    """Raised when a social-intelligence safety or data contract fails."""


@dataclass(frozen=True)
class SocialConfig:
    database_path: Path
    state_path: Path
    api_base_url: str
    request_timeout_seconds: float
    max_posts_per_run: int
    sorts: tuple[str, ...]
    submolts: tuple[str, ...]
    search_queries: tuple[str, ...]
    relevance_terms: tuple[str, ...]
    publication_enabled: bool
    publication_submolts: tuple[str, ...]
    api_key_env: str
    secret_env_file: Path

    @classmethod
    def load(cls, path: str | Path) -> "SocialConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != CONFIG_SCHEMA:
            raise SocialIntelligenceError("Unsupported social-intelligence config schema")
        api_base = str(payload.get("api_base_url", "")).rstrip("/")
        if api_base != MOLTBOOK_BASE_URL:
            raise SocialIntelligenceError(
                "Moltbook credentials and requests are restricted to the official www API"
            )
        collection = payload.get("collection") or {}
        publishing = payload.get("publishing") or {}
        max_posts = int(collection.get("max_posts_per_run", 100))
        if not 1 <= max_posts <= 500:
            raise SocialIntelligenceError("max_posts_per_run must be between 1 and 500")
        sorts = tuple(str(item) for item in collection.get("sorts", ["new"]))
        if not sorts or any(item not in {"new", "hot", "top", "rising"} for item in sorts):
            raise SocialIntelligenceError("Unsupported or empty Moltbook sort list")
        allowed_submolts = tuple(
            str(item).strip().lower()
            for item in collection.get("submolts", [])
            if str(item).strip()
        )
        publication_submolts = tuple(
            str(item).strip().lower()
            for item in publishing.get("allowed_submolts", [])
            if str(item).strip()
        )
        if publishing.get("enabled") and not publication_submolts:
            raise SocialIntelligenceError(
                "Publishing requires an explicit non-empty submolt allowlist"
            )
        secrets = payload.get("secrets") or {}
        return cls(
            database_path=expand_path(
                str(payload.get("database_path", "~/.local/state/agent-multi/social-intelligence.sqlite"))
            ),
            state_path=expand_path(
                str(payload.get("state_path", "~/.local/state/agent-multi/social-intelligence-state.json"))
            ),
            api_base_url=api_base,
            request_timeout_seconds=float(payload.get("request_timeout_seconds", 20)),
            max_posts_per_run=max_posts,
            sorts=sorts,
            submolts=allowed_submolts,
            search_queries=tuple(
                str(item).strip()
                for item in collection.get("search_queries", [])
                if str(item).strip()
            ),
            relevance_terms=tuple(
                str(item).strip().lower()
                for item in collection.get("relevance_terms", [])
                if str(item).strip()
            ),
            publication_enabled=bool(publishing.get("enabled", False)),
            publication_submolts=publication_submolts,
            api_key_env=str(secrets.get("api_key_env", "MOLTBOOK_API_KEY")),
            secret_env_file=expand_path(
                str(secrets.get("env_file", "~/.config/agent-multi/moltbook.env"))
            ),
        )


class MoltbookClient:
    """Small API client that never forwards the API key outside the official host."""

    def __init__(
        self,
        *,
        api_base_url: str = MOLTBOOK_BASE_URL,
        api_key: str = "",
        timeout_seconds: float = 20,
        opener: Any = None,
    ) -> None:
        if api_base_url.rstrip("/") != MOLTBOOK_BASE_URL:
            raise SocialIntelligenceError("Refusing a non-official Moltbook API base URL")
        self.base_url = MOLTBOOK_BASE_URL
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.opener = opener or urllib.request.urlopen

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
        require_auth: bool = False,
    ) -> Mapping[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise SocialIntelligenceError("Invalid Moltbook API path")
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        if not url.startswith(MOLTBOOK_BASE_URL + "/"):
            raise SocialIntelligenceError("Moltbook request escaped the approved API prefix")
        if require_auth and not self.api_key:
            raise SocialIntelligenceError("MOLTBOOK_API_KEY is required for this operation")
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = canonical_json(body).encode("utf-8") if body is not None else None
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            response = self.opener(request, timeout=self.timeout_seconds)
            raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(400).decode("utf-8", errors="replace")
            raise SocialIntelligenceError(
                f"Moltbook HTTP {exc.code}: {detail[:240]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SocialIntelligenceError(
                f"Moltbook request failed: {type(exc.reason).__name__}"
            ) from exc
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise SocialIntelligenceError("Moltbook returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise SocialIntelligenceError("Moltbook response must be a JSON object")
        if payload.get("success") is False:
            raise SocialIntelligenceError(
                f"Moltbook rejected the request: {payload.get('error') or payload.get('message')}"
            )
        return payload

    def posts(self, *, sort: str, limit: int, submolt: str | None = None) -> list[Mapping[str, Any]]:
        params: dict[str, Any] = {"sort": sort, "limit": limit}
        if submolt:
            params["submolt"] = submolt
        payload = self._request("GET", "/posts", params=params)
        return [item for item in payload.get("posts", []) if isinstance(item, Mapping)]

    def search(self, query: str, *, limit: int) -> list[Mapping[str, Any]]:
        payload = self._request(
            "GET",
            "/search",
            params={"q": query, "type": "posts", "limit": limit},
        )
        values = payload.get("results") or payload.get("posts") or []
        return [item for item in values if isinstance(item, Mapping)]

    def me(self) -> Mapping[str, Any]:
        return self._request("GET", "/agents/me", require_auth=True)

    def create_post(self, *, submolt: str, title: str, content: str) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/posts",
            body={"submolt_name": submolt, "title": title, "content": content, "type": "text"},
            require_auth=True,
        )

    def verify_content(self, *, verification_code: str, answer: str) -> Mapping[str, Any]:
        if not re.fullmatch(r"-?\d+(?:\.\d{2})", answer):
            raise SocialIntelligenceError(
                "Verification answer must contain exactly two decimal places"
            )
        return self._request(
            "POST",
            "/verify",
            body={"verification_code": verification_code, "answer": answer},
            require_auth=True,
        )


def injection_flags(text: str) -> list[str]:
    return [f"pattern_{index + 1}" for index, pattern in enumerate(INJECTION_PATTERNS) if pattern.search(text)]


def normalize_post(raw: Mapping[str, Any], *, retrieval_source: str) -> dict[str, Any] | None:
    external_id = str(raw.get("id") or raw.get("post_id") or "").strip()
    if not external_id:
        return None
    title = str(raw.get("title") or "").strip()
    content = str(raw.get("content") or raw.get("text") or "").strip()
    combined = f"{title}\n{content}".strip()
    submolt = raw.get("submolt_name") or raw.get("submolt")
    if isinstance(submolt, Mapping):
        submolt = submolt.get("name")
    author = raw.get("author_name") or raw.get("agent_name") or raw.get("author")
    if isinstance(author, Mapping):
        author = author.get("name")
    url = str(raw.get("url") or f"https://www.moltbook.com/post/{external_id}")
    if url.startswith("/"):
        url = f"https://www.moltbook.com{url}"
    if not url.startswith("https://www.moltbook.com/"):
        url = f"https://www.moltbook.com/post/{external_id}"
    return {
        "external_id": external_id,
        "retrieval_source": retrieval_source,
        "submolt": str(submolt or "unknown").lower(),
        "author": str(author or "unknown"),
        "title": title[:300],
        "content": content[:40000],
        "url": url,
        "published_at": raw.get("created_at") or raw.get("published_at"),
        "retrieved_at": utc_now(),
        "content_sha256": sha256_text(combined),
        "injection_flags": injection_flags(combined),
        "raw_sha256": sha256_text(canonical_json(raw)),
    }


def relevance_score(post: Mapping[str, Any], terms: Sequence[str]) -> float:
    text = f"{post.get('title', '')}\n{post.get('content', '')}".lower()
    if not terms:
        return 0.0
    matches = sum(1 for term in terms if term in text)
    return min(1.0, matches / max(3.0, len(terms) ** 0.5))


class SocialOlap:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self._initialize()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS collection_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                status TEXT NOT NULL,
                fetched INTEGER NOT NULL DEFAULT 0,
                inserted INTEGER NOT NULL DEFAULT 0,
                duplicates INTEGER NOT NULL DEFAULT 0,
                flagged INTEGER NOT NULL DEFAULT 0,
                error_kind TEXT
            );
            CREATE TABLE IF NOT EXISTS posts (
                external_id TEXT PRIMARY KEY,
                retrieval_source TEXT NOT NULL,
                submolt TEXT NOT NULL,
                author TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                url TEXT NOT NULL,
                published_at TEXT,
                first_retrieved_at TEXT NOT NULL,
                last_retrieved_at TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                raw_sha256 TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                injection_flags_json TEXT NOT NULL,
                review_state TEXT NOT NULL DEFAULT 'unreviewed'
            );
            CREATE INDEX IF NOT EXISTS posts_relevance_idx
                ON posts(review_state,relevance_score DESC,last_retrieved_at DESC);
            CREATE TABLE IF NOT EXISTS drafts (
                draft_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                submolt TEXT NOT NULL,
                source_post_ids_json TEXT NOT NULL,
                source_hashes_json TEXT NOT NULL,
                state TEXT NOT NULL,
                approved_at TEXT,
                approved_by TEXT,
                published_at TEXT,
                external_post_id TEXT,
                response_sha256 TEXT
            );
            CREATE TABLE IF NOT EXISTS digest_runs (
                digest_id TEXT PRIMARY KEY,
                generated_at TEXT NOT NULL,
                cutoff_at TEXT NOT NULL,
                selected_post_ids_json TEXT NOT NULL,
                packet_sha256 TEXT NOT NULL
            );
            CREATE VIEW IF NOT EXISTS social_source_yield_olap AS
            SELECT retrieval_source,submolt,COUNT(*) AS posts,
                   SUM(CASE WHEN relevance_score >= 0.25 THEN 1 ELSE 0 END) AS relevant_posts,
                   SUM(CASE WHEN injection_flags_json != '[]' THEN 1 ELSE 0 END) AS flagged_posts,
                   MAX(last_retrieved_at) AS last_retrieved_at
            FROM posts GROUP BY retrieval_source,submolt;
            """
        )
        columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(drafts)")
        }
        for name in (
            "verification_code",
            "verification_challenge",
            "verification_expires_at",
        ):
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE drafts ADD COLUMN {name} TEXT"
                )
        self.connection.commit()

    def start_run(self) -> str:
        run_id = f"collect-{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            "INSERT INTO collection_runs(run_id,started_at,status) VALUES (?,?,'running')",
            (run_id, utc_now()),
        )
        self.connection.commit()
        return run_id

    def ingest(self, post: Mapping[str, Any], score: float) -> bool:
        existing = self.connection.execute(
            "SELECT external_id FROM posts WHERE external_id=?",
            (post["external_id"],),
        ).fetchone()
        if existing:
            self.connection.execute(
                """
                UPDATE posts SET last_retrieved_at=?,retrieval_source=?,
                    relevance_score=MAX(relevance_score,?)
                WHERE external_id=?
                """,
                (
                    post["retrieved_at"],
                    post["retrieval_source"],
                    score,
                    post["external_id"],
                ),
            )
            return False
        self.connection.execute(
            """
            INSERT INTO posts(
                external_id,retrieval_source,submolt,author,title,content,url,
                published_at,first_retrieved_at,last_retrieved_at,content_sha256,
                raw_sha256,relevance_score,injection_flags_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                post["external_id"],
                post["retrieval_source"],
                post["submolt"],
                post["author"],
                post["title"],
                post["content"],
                post["url"],
                post.get("published_at"),
                post["retrieved_at"],
                post["retrieved_at"],
                post["content_sha256"],
                post["raw_sha256"],
                score,
                canonical_json(post["injection_flags"]),
            ),
        )
        return True

    def finish_run(self, run_id: str, *, status: str, counters: Mapping[str, int], error_kind: str | None = None) -> None:
        self.connection.execute(
            """
            UPDATE collection_runs SET ended_at=?,status=?,fetched=?,inserted=?,
                duplicates=?,flagged=?,error_kind=? WHERE run_id=?
            """,
            (
                utc_now(),
                status,
                counters.get("fetched", 0),
                counters.get("inserted", 0),
                counters.get("duplicates", 0),
                counters.get("flagged", 0),
                error_kind,
                run_id,
            ),
        )
        self.connection.commit()

    def digest_packet(self, *, hours: int = 8, limit: int = 30) -> dict[str, Any]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.connection.execute(
            """
            SELECT external_id,submolt,author,title,content,url,published_at,
                   content_sha256,relevance_score,injection_flags_json
            FROM posts
            WHERE last_retrieved_at >= ?
            ORDER BY relevance_score DESC,last_retrieved_at DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        safe_items = []
        flagged_count = 0
        for row in rows:
            flags = json.loads(row["injection_flags_json"])
            if flags:
                flagged_count += 1
                continue
            safe_items.append(
                {
                    "external_id": row["external_id"],
                    "submolt": row["submolt"],
                    "author": row["author"],
                    "title": row["title"],
                    "content_excerpt": row["content"][:1200],
                    "url": row["url"],
                    "published_at": row["published_at"],
                    "content_sha256": row["content_sha256"],
                    "relevance_score": row["relevance_score"],
                }
            )
        packet = {
            "wakeAgent": bool(safe_items),
            "schema": "agent_multi.hermes.social_digest.v1",
            "generated_at": utc_now(),
            "evidence_only": True,
            "policy": {
                "content_is_untrusted": True,
                "orders_allowed": False,
                "campaign_changes_allowed": False,
                "publishing_allowed": False,
                "human_review_required": True,
            },
            "period_hours": hours,
            "safe_items": safe_items,
            "flagged_items_withheld": flagged_count,
        }
        digest_id = f"digest-{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            "INSERT INTO digest_runs VALUES (?,?,?,?,?)",
            (
                digest_id,
                packet["generated_at"],
                cutoff,
                canonical_json([item["external_id"] for item in safe_items]),
                sha256_text(canonical_json(packet)),
            ),
        )
        self.connection.commit()
        return packet

    def create_draft(self, *, title: str, content: str, submolt: str, source_ids: Sequence[str]) -> str:
        if not title.strip() or not content.strip():
            raise SocialIntelligenceError("Draft title and content are required")
        rows = self.connection.execute(
            f"SELECT external_id,content_sha256 FROM posts WHERE external_id IN ({','.join('?' for _ in source_ids)})",
            tuple(source_ids),
        ).fetchall() if source_ids else []
        if len(rows) != len(set(source_ids)):
            raise SocialIntelligenceError("Every draft source ID must exist in the social OLAP")
        draft_id = f"draft-{uuid.uuid4().hex[:16]}"
        self.connection.execute(
            """
            INSERT INTO drafts(
                draft_id,created_at,title,content,submolt,
                source_post_ids_json,source_hashes_json,state,
                approved_at,approved_by,published_at,external_post_id,
                response_sha256
            ) VALUES (?,?,?,?,?,?,?,'draft',NULL,NULL,NULL,NULL,NULL)
            """,
            (
                draft_id,
                utc_now(),
                title.strip()[:300],
                content.strip()[:40000],
                submolt.strip().lower(),
                canonical_json(sorted(set(source_ids))),
                canonical_json({row["external_id"]: row["content_sha256"] for row in rows}),
            ),
        )
        self.connection.commit()
        return draft_id

    def approve_draft(self, draft_id: str, approved_by: str) -> None:
        changed = self.connection.execute(
            """
            UPDATE drafts SET state='approved',approved_at=?,approved_by=?
            WHERE draft_id=? AND state='draft'
            """,
            (utc_now(), approved_by, draft_id),
        ).rowcount
        self.connection.commit()
        if changed != 1:
            raise SocialIntelligenceError("Draft is missing or is not awaiting approval")

    def approved_drafts(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM drafts WHERE state='approved' ORDER BY approved_at"
        ).fetchall()

    def mark_post_created(self, draft_id: str, response: Mapping[str, Any]) -> str:
        post = response.get("post") if isinstance(response.get("post"), Mapping) else response
        external_id = str(post.get("id") or post.get("post_id") or "")
        verification = (
            post.get("verification")
            if isinstance(post.get("verification"), Mapping)
            else {}
        )
        requires_verification = bool(
            response.get("verification_required")
            or post.get("verification_status") == "pending"
            or verification
        )
        state = "verification_pending" if requires_verification else "published"
        self.connection.execute(
            """
            UPDATE drafts SET state=?,published_at=?,external_post_id=?,
                response_sha256=?,verification_code=?,verification_challenge=?,
                verification_expires_at=?
            WHERE draft_id=? AND state='approved'
            """,
            (
                state,
                utc_now() if state == "published" else None,
                external_id,
                sha256_text(canonical_json(response)),
                verification.get("verification_code"),
                verification.get("challenge_text"),
                verification.get("expires_at"),
                draft_id,
            ),
        )
        self.connection.commit()
        return state

    def pending_verification(self, draft_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            """
            SELECT * FROM drafts
            WHERE draft_id=? AND state='verification_pending'
            """,
            (draft_id,),
        ).fetchone()
        if row is None:
            raise SocialIntelligenceError(
                "Draft is not awaiting Moltbook verification"
            )
        return row

    def mark_verified(
        self,
        draft_id: str,
        response: Mapping[str, Any],
    ) -> None:
        changed = self.connection.execute(
            """
            UPDATE drafts SET state='published',published_at=?,response_sha256=?
            WHERE draft_id=? AND state='verification_pending'
            """,
            (
                utc_now(),
                sha256_text(canonical_json(response)),
                draft_id,
            ),
        ).rowcount
        self.connection.commit()
        if changed != 1:
            raise SocialIntelligenceError(
                "Draft verification state changed unexpectedly"
            )

    def status(self) -> dict[str, Any]:
        counts = dict(
            self.connection.execute(
                "SELECT review_state,COUNT(*) FROM posts GROUP BY review_state"
            ).fetchall()
        )
        drafts = dict(
            self.connection.execute(
                "SELECT state,COUNT(*) FROM drafts GROUP BY state"
            ).fetchall()
        )
        latest = self.connection.execute(
            "SELECT * FROM collection_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return {
            "schema": SCHEMA_VERSION,
            "database_path": str(self.path),
            "posts_by_review_state": counts,
            "drafts_by_state": drafts,
            "latest_collection": dict(latest) if latest else None,
        }


def collect(config: SocialConfig, client: MoltbookClient, store: SocialOlap) -> dict[str, Any]:
    run_id = store.start_run()
    counters = {"fetched": 0, "inserted": 0, "duplicates": 0, "flagged": 0}
    seen: set[str] = set()
    try:
        sources: list[tuple[str, Iterable[Mapping[str, Any]]]] = []
        per_source = max(5, min(100, config.max_posts_per_run // max(1, len(config.sorts) + len(config.submolts) + len(config.search_queries))))
        for sort in config.sorts:
            sources.append((f"feed:{sort}", client.posts(sort=sort, limit=per_source)))
        for submolt in config.submolts:
            sources.append(
                (f"submolt:{submolt}", client.posts(sort="new", limit=per_source, submolt=submolt))
            )
        for query in config.search_queries:
            sources.append((f"search:{query}", client.search(query, limit=per_source)))
        for source, values in sources:
            for raw in values:
                post = normalize_post(raw, retrieval_source=source)
                if not post or post["external_id"] in seen:
                    continue
                seen.add(post["external_id"])
                counters["fetched"] += 1
                counters["flagged"] += int(bool(post["injection_flags"]))
                if store.ingest(post, relevance_score(post, config.relevance_terms)):
                    counters["inserted"] += 1
                else:
                    counters["duplicates"] += 1
                if counters["fetched"] >= config.max_posts_per_run:
                    break
            if counters["fetched"] >= config.max_posts_per_run:
                break
        store.finish_run(run_id, status="complete", counters=counters)
        return {"run_id": run_id, "status": "complete", **counters}
    except Exception as exc:
        store.finish_run(
            run_id,
            status="failed",
            counters=counters,
            error_kind=type(exc).__name__,
        )
        raise


def publish_approved(config: SocialConfig, client: MoltbookClient, store: SocialOlap) -> dict[str, Any]:
    if not config.publication_enabled:
        raise SocialIntelligenceError("Publishing is disabled in the tracked config")
    published: list[str] = []
    for draft in store.approved_drafts():
        if draft["submolt"] not in config.publication_submolts:
            raise SocialIntelligenceError(
                f"Draft {draft['draft_id']} targets a non-allowlisted submolt"
            )
        response = client.create_post(
            submolt=draft["submolt"],
            title=draft["title"],
            content=draft["content"],
        )
        state = store.mark_post_created(draft["draft_id"], response)
        published.append(draft["draft_id"])
        if state == "verification_pending":
            post = (
                response.get("post")
                if isinstance(response.get("post"), Mapping)
                else {}
            )
            return {
                "published": [],
                "verification_pending": [
                    {
                        "draft_id": draft["draft_id"],
                        "challenge": (
                            post.get("verification") or {}
                        ).get("challenge_text"),
                        "expires_at": (
                            post.get("verification") or {}
                        ).get("expires_at"),
                    }
                ],
                "count": 0,
            }
        break
    return {"published": published, "count": len(published)}


def verify_pending(
    client: MoltbookClient,
    store: SocialOlap,
    *,
    draft_id: str,
    answer: str,
) -> dict[str, Any]:
    draft = store.pending_verification(draft_id)
    response = client.verify_content(
        verification_code=draft["verification_code"],
        answer=answer,
    )
    store.mark_verified(draft_id, response)
    return {"verified": draft_id, "external_post_id": draft["external_post_id"]}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("collect")
    digest = sub.add_parser("digest-context")
    digest.add_argument("--hours", type=int, default=8)
    digest.add_argument("--limit", type=int, default=30)
    sub.add_parser("status")
    draft = sub.add_parser("create-draft")
    draft.add_argument("--title", required=True)
    draft.add_argument("--content-file", required=True, type=Path)
    draft.add_argument("--submolt", required=True)
    draft.add_argument("--source-id", action="append", default=[])
    approve = sub.add_parser("approve")
    approve.add_argument("draft_id")
    approve.add_argument("--approved-by", default="human-owner")
    sub.add_parser("publish-approved")
    verify = sub.add_parser("verify-draft")
    verify.add_argument("draft_id")
    verify.add_argument("--answer", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = SocialConfig.load(args.config)
    api_key = load_secret_env_value(config.api_key_env, config.secret_env_file)
    client = MoltbookClient(
        api_base_url=config.api_base_url,
        api_key=api_key,
        timeout_seconds=config.request_timeout_seconds,
    )
    store = SocialOlap(config.database_path)
    try:
        if args.command == "collect":
            result = collect(config, client, store)
        elif args.command == "digest-context":
            result = store.digest_packet(hours=args.hours, limit=args.limit)
        elif args.command == "status":
            result = store.status()
            if api_key:
                try:
                    profile = client.me()
                    agent = profile.get("agent") if isinstance(profile.get("agent"), Mapping) else profile
                    result["moltbook_identity"] = {
                        "available": True,
                        "name": agent.get("name"),
                        "claimed": agent.get("is_claimed") or agent.get("status") == "claimed",
                    }
                except SocialIntelligenceError as exc:
                    result["moltbook_identity"] = {"available": False, "reason": str(exc)}
            else:
                result["moltbook_identity"] = {"available": False, "reason": "credential_missing"}
        elif args.command == "create-draft":
            result = {
                "draft_id": store.create_draft(
                    title=args.title,
                    content=args.content_file.read_text(encoding="utf-8"),
                    submolt=args.submolt,
                    source_ids=args.source_id,
                )
            }
        elif args.command == "approve":
            store.approve_draft(args.draft_id, args.approved_by)
            result = {"approved": args.draft_id}
        elif args.command == "publish-approved":
            result = publish_approved(config, client, store)
        elif args.command == "verify-draft":
            result = verify_pending(
                client,
                store,
                draft_id=args.draft_id,
                answer=args.answer,
            )
        else:
            raise AssertionError(args.command)
    except SocialIntelligenceError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    finally:
        store.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
