#!/usr/bin/env python3
"""Emit a sanitized, budget-reserved social evidence packet for Hermes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from social_intelligence import (
    SocialConfig,
    SocialIntelligenceError,
    SocialOlap,
    canonical_json,
    sha256_text,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--tier", required=True)
    value.add_argument("--provider", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--prompt-template-sha256", required=True)
    value.add_argument("--hours", type=int, default=8)
    value.add_argument("--limit", type=int, default=30)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        config = SocialConfig.load(args.config)
        store = SocialOlap(config.database_path)
        try:
            packet = store.digest_packet(hours=args.hours, limit=args.limit)
            if packet["wakeAgent"]:
                packet_json = canonical_json(packet)
                budget = store.reserve_model_call(
                    config,
                    tier=args.tier,
                    provider=args.provider,
                    model=args.model,
                    prompt_template_sha256=args.prompt_template_sha256,
                    packet_sha256=sha256_text(packet_json),
                    input_chars=len(packet_json),
                )
                packet["model_budget"] = budget
                if budget["status"] != "reserved":
                    packet["wakeAgent"] = False
                    packet["reason"] = budget["block_reason"]
            else:
                packet["model_budget"] = {
                    "status": "not_reserved",
                    "reason": "no_safe_items",
                }
        finally:
            store.close()
    except (OSError, ValueError, SocialIntelligenceError):
        print('{"wakeAgent":false,"reason":"social_context_unavailable"}')
        return 0
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
