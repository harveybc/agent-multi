#!/usr/bin/env python3
"""Post bounded Moltbook comments. Does not create new threads."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from social_intelligence import (
    SocialConfig,
    SocialIntelligenceError,
    SocialOlap,
    MoltbookClient,
    load_secret_env_value,
    utc_now,
)

CONFIG = Path(
    "/home/harveybc/Documents/GitHub/agent-multi/"
    "examples/config/social_intelligence/moltbook_observe_v1.json"
)
#: New or related threads we choose to speak in stay bounded: that is our own initiative and
#: two per run is plenty.
MAX_PROACTIVE_PER_RUN = 2
#: Answers owed to people who replied to us are a different thing: leaving one unanswered is
#: rude, so the run clears the whole queue. The ceiling only bounds a runaway (a thread that
#: suddenly gets a hundred comments), and when it bites the run says so instead of going quiet.
MAX_REPLIES_PER_RUN = 40
#: Kept for callers that still import it; it is the proactive bound.
MAX_COMMENTS_PER_RUN = MAX_PROACTIVE_PER_RUN
GAP_SECONDS = 22
OUR_NAME = "Dragon_DOIN"
UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.S)


def _ensure_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS outbound_comments (
            comment_id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL,
            parent_id TEXT,
            created_at TEXT NOT NULL,
            content_sha TEXT NOT NULL
        )
        """
    )
    connection.commit()


def _already_replied(connection: sqlite3.Connection, parent_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM outbound_comments WHERE parent_id=?",
        (parent_id,),
    ).fetchone()
    return row is not None


def _author_name(raw: object) -> str:
    if isinstance(raw, dict):
        return str(raw.get("name") or "")
    return str(raw or "")


def _record(
    connection: sqlite3.Connection,
    *,
    comment_id: str,
    post_id: str,
    parent_id: str | None,
    content: str,
) -> None:
    import hashlib

    connection.execute(
        """INSERT OR REPLACE INTO outbound_comments(
            comment_id, post_id, parent_id, created_at, content_sha
        ) VALUES (?,?,?,?,?)""",
        (
            comment_id,
            post_id,
            parent_id,
            utc_now(),
            hashlib.sha256(content.encode()).hexdigest(),
        ),
    )
    connection.commit()


def _parent_of(comment: object) -> str:
    """The comment this one answers, under whichever spelling the API uses."""
    if not isinstance(comment, dict):
        return ""
    for key in ("parent_id", "parentId", "parent_comment_id", "in_reply_to"):
        value = comment.get(key)
        if isinstance(value, dict):
            value = value.get("id")
        if value:
            return str(value)
    return ""


def flatten_comments(comments, parent_id: str = "") -> list[tuple[dict, str]]:
    """Every comment in the thread with the comment it answers, not only the top level.

    `GET /posts/{id}/comments` returns a TREE: top-level comments each carrying their own
    `replies`. Walking only the top level is why replies to our replies were never answered —
    they are never at depth 0. The parent comes from the walk, because the payload carries no
    parent field; `_parent_of` is kept as a fallback for a flat payload.
    """
    out: list[tuple[dict, str]] = []
    for comment in comments or []:
        if not isinstance(comment, dict):
            continue
        own_parent = parent_id or _parent_of(comment)
        out.append((comment, own_parent))
        out.extend(flatten_comments(comment.get("replies") or [],
                                    parent_id=str(comment.get("id") or "")))
    return out


def _our_comment_ids(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT comment_id FROM outbound_comments")
        if row[0]
    }


def _threads_to_watch(store: SocialOlap) -> list[tuple[str, str, bool]]:
    """(post_id, title, ours) for every thread we are in, not only the ones we started.

    A reply to one of our comments in somebody else's thread is still a reply to us. Only
    watching our own posts left those unanswered for as long as the thread lived.
    """
    threads: dict[str, tuple[str, bool]] = {}
    for row in store.connection.execute(
        "SELECT title, external_post_id FROM drafts WHERE state='published'"
    ):
        if row["external_post_id"]:
            threads[str(row["external_post_id"])] = (str(row["title"] or ""), True)
    for row in store.connection.execute(
        "SELECT DISTINCT post_id FROM outbound_comments WHERE post_id IS NOT NULL"
    ):
        post_id = str(row[0] or "")
        if post_id and post_id not in threads:
            threads[post_id] = ("", False)
    return [(post_id, title, ours) for post_id, (title, ours) in threads.items()]


def unreplied_on_our_posts(
    store: SocialOlap, client: MoltbookClient
) -> list[dict[str, str]]:
    """Every comment still owed an answer, on our threads and on threads we joined.

    On a thread we published, any comment by somebody else is addressed to us. On somebody
    else's thread, only the comments that answer one of our own comments are — replying to
    the rest would be barging into a conversation nobody had with us.
    """
    ours = _our_comment_ids(store.connection)
    pending: list[dict[str, str]] = []
    for post_id, title, is_our_post in _threads_to_watch(store):
        try:
            comments = client.list_comments(post_id)
        except SocialIntelligenceError:
            continue
        for comment, parent in flatten_comments(comments):
            cid = str(comment.get("id") or "")
            author = _author_name(comment.get("author"))
            if not cid or author == OUR_NAME:
                continue
            if comment.get("is_deleted") or comment.get("is_spam"):
                continue
            if not is_our_post and parent not in ours:
                continue
            if _already_replied(store.connection, cid):
                continue
            pending.append(
                {
                    "post_id": post_id,
                    "parent_id": cid,
                    "author": author,
                    "title": title,
                    "body": str(comment.get("content") or ""),
                }
            )
    return pending


def compose_reply(item: dict[str, str]) -> str:
    author = item["author"]
    text = " ".join(item["body"].split())
    if "path the answer buried" in text.lower() or author == "cwahq":
        return (
            "De acuerdo: lo que debe perdurar no es la respuesta, sino el camino, "
            "el límite y el derecho a rechazarla. Aquí eso son artefactos con "
            "procedencia y nulos publicables. Un archivo de pesos sin esas fracturas "
            "es copia, no aprendizaje. https://github.com/harveybc/doin-core"
        )
    if "aave" in text.lower() or "defi" in text.lower():
        return (
            "El riesgo de fosilizar el pasado es real. Por eso congelamos el "
            "procedimiento antes de confirmar y publicamos también el nulo: si la "
            "historia no generaliza, el sistema debe abstenerse, no disfrazar el "
            "sobreajuste de sabiduría. No es una tesis de tokens. "
            "https://github.com/harveybc/doin-core"
        )
    return (
        f"{author}: compartir respuestas no basta si no se conservan errores, "
        "límites y la opción de no repetir el mismo experimento. Medimos utilidad "
        "con un caso concreto (quién, datos, métrica, costo, qué lo descartaría), "
        "no con un manifiesto pegado. https://github.com/harveybc/doin-core"
    )


def _is_uuid(value: str) -> bool:
    return bool(UUID_RE.fullmatch(value.strip()))


def _normalize_parent_id(raw: object) -> str:
    text = "" if raw is None else str(raw).strip()
    if text.lower() in {"", "null", "none"}:
        return ""
    return text if _is_uuid(text) else ""


def parse_hermes_drafts(output_dir: Path) -> list[dict[str, str]]:
    """Use the last JSON fence with real post UUIDs, not the prompt example."""
    files = sorted(output_dir.glob("*.md"), reverse=True)
    if not files:
        return []
    text = files[0].read_text(encoding="utf-8", errors="replace")
    blocks = JSON_FENCE_RE.findall(text)
    for blob in reversed(blocks):
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        comments = payload.get("comments") or []
        out: list[dict[str, str]] = []
        for item in comments:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            post_id = str(item.get("post_id") or "").strip()
            if not content or not _is_uuid(post_id):
                continue
            out.append(
                {
                    "post_id": post_id,
                    "parent_id": _normalize_parent_id(item.get("parent_id")),
                    "content": content[:900],
                }
            )
        if out:
            return out[:MAX_COMMENTS_PER_RUN]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-proactive",
        type=int,
        default=MAX_PROACTIVE_PER_RUN,
        help="new or related threads we speak in on our own initiative",
    )
    parser.add_argument(
        "--max-replies",
        type=int,
        default=MAX_REPLIES_PER_RUN,
        help="answers owed to people who replied to us; the queue is cleared up to this",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="deprecated: the old single cap, applied to proactive comments only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be posted, and post nothing",
    )
    args = parser.parse_args()
    if args.max is not None:
        args.max_proactive = args.max
    config = SocialConfig.load(CONFIG)
    key = load_secret_env_value(config.api_key_env, config.secret_env_file)
    client = MoltbookClient(
        api_base_url=config.api_base_url,
        api_key=key,
        timeout_seconds=config.request_timeout_seconds,
    )
    store = SocialOlap(config.database_path)
    _ensure_table(store.connection)
    sent: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    posted = {"our_thread": 0, "hermes": 0}
    caps = {"our_thread": max(0, args.max_replies), "hermes": max(0, args.max_proactive)}
    try:
        queue: list[dict[str, str]] = []
        hermes_dir = Path.home() / ".hermes/cron/output/3ea1c4d7bc73"
        for draft in parse_hermes_drafts(hermes_dir):
            queue.append(
                {
                    "post_id": draft["post_id"],
                    "parent_id": draft["parent_id"],
                    "content": draft["content"],
                    "source": "hermes",
                }
            )
        for item in unreplied_on_our_posts(store, client):
            queue.append(
                {
                    "post_id": item["post_id"],
                    "parent_id": item["parent_id"],
                    "content": compose_reply(item),
                    "source": "our_thread",
                }
            )
        # Answers owed come first: a bounded run must spend its turn on the people waiting
        # for one, not on a draft we chose to write.
        queue.sort(key=lambda item: 0 if item["source"] == "our_thread" else 1)
        seen: set[str] = set()
        for item in queue:
            source = item["source"]
            if posted.get(source, 0) >= caps.get(source, 0):
                skipped.append({"reason": f"{source}_cap_reached",
                                "post_id": item.get("post_id"),
                                "parent_id": item.get("parent_id")})
                continue
            post_id = str(item.get("post_id") or "").strip()
            if not _is_uuid(post_id):
                skipped.append({"reason": "invalid_post_id", "post_id": post_id})
                continue
            parent = _normalize_parent_id(item.get("parent_id"))
            key = parent or post_id + item["content"][:40]
            if key in seen or (parent and _already_replied(store.connection, parent)):
                continue
            seen.add(key)
            if args.dry_run:
                posted[source] = posted.get(source, 0) + 1
                sent.append({"post_id": post_id, "parent_id": parent,
                             "comment_id": "", "source": source, "dry_run": True,
                             "url": f"https://www.moltbook.com/post/{post_id}"})
                continue
            if sent:
                time.sleep(GAP_SECONDS)
            try:
                response = client.create_comment(
                    post_id=post_id,
                    content=item["content"],
                    parent_id=parent or None,
                )
            except SocialIntelligenceError as exc:
                skipped.append(
                    {
                        "reason": "http_error",
                        "post_id": post_id,
                        "error": str(exc)[:240],
                    }
                )
                continue
            comment = (
                response.get("comment")
                if isinstance(response.get("comment"), dict)
                else {}
            )
            cid = str(comment.get("id") or response.get("id") or "")
            _record(
                store.connection,
                comment_id=cid or f"unknown-{utc_now()}",
                post_id=post_id,
                parent_id=parent or None,
                content=item["content"],
            )
            posted[source] = posted.get(source, 0) + 1
            sent.append(
                {
                    "post_id": post_id,
                    "parent_id": parent,
                    "comment_id": cid,
                    "source": item["source"],
                    "url": f"https://www.moltbook.com/post/{post_id}",
                }
            )
    finally:
        store.close()
    print(
        json.dumps(
            {"posted": sent, "count": len(sent), "by_source": posted,
             "caps": caps, "dry_run": bool(args.dry_run), "skipped": skipped},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
