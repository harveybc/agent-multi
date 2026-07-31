from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from validate_incident_corpus_manifest import validate_manifest  # noqa: E402


def test_incident_corpus_enumeration_rule_is_hash_pinned() -> None:
    result = validate_manifest()

    assert result["status"] == "valid"
    assert result["rule_sha256"].startswith("6abc241d")
