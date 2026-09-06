"""PRE freeze for order C26: amendment 9 was edited in place
between d97c3f62 and d8f25438 — proven from public Git objects.
Kept as a regression: it fails if either Git object is silently
substituted."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
A9_REL = ("docs/audits/evidence/"
          "B4_SUPERSEDING_DESIGN_V2_AMENDMENT_9_2026_09_06.json")
WANT_ORIGINAL = ("eb9d49707b2a173056b07c3802b617d8302b38e5f42245ef"
                 "ed7db5e8d155ca42")
WANT_REWRITTEN = ("01aeee957c993ee764b8e4753860d9c03b6467973eed5a3"
                  "ec07bb425e1e4a337")


def blob_at(commit: str) -> bytes:
    return subprocess.run(
        ["git", "cat-file", "-p", f"{commit}:{A9_REL}"],
        cwd=REPO, capture_output=True, check=True).stdout


orig = blob_at("d97c3f62")
rewr = blob_at("d8f25438")
h_orig = hashlib.sha256(orig).hexdigest()
h_rewr = hashlib.sha256(rewr).hexdigest()
print("amendment 9 @ d97c3f62:", h_orig)
print("amendment 9 @ d8f25438:", h_rewr)
assert h_orig == WANT_ORIGINAL, "git object substituted (original)"
assert h_rewr == WANT_REWRITTEN, "git object substituted (rewritten)"
assert h_orig != h_rewr

d_orig = json.loads(orig)
d_rewr = json.loads(rewr)
diff_keys = [k for k in sorted(set(d_orig) | set(d_rewr))
             if d_orig.get(k) != d_rewr.get(k)]
print("differing top-level fields:", diff_keys)
assert diff_keys == ["final_code_pins"], (
    "expected ONLY the code pins to differ")
pin_diff = sorted(
    rel for rel in set(d_orig["final_code_pins"])
    | set(d_rewr["final_code_pins"])
    if d_orig["final_code_pins"].get(rel)
    != d_rewr["final_code_pins"].get(rel))
print("differing pins:", pin_diff)
assert set(pin_diff) == {
    "tools/b4_campaign_executor.py",
    "tools/b4_campaign_ledger.py",
    "tools/b4_campaign_orchestrator.py",
    "tests/test_b4_materializer_authority.py",
}, "differences are not the C23-C25 runtime/test surface"
print("-> the supposedly append-only amendment 9 was edited in "
      "place to carry the C23-C25 pins; nothing else changed")
print("\nPRE CONFIRMED: C26 finding reproduces from Git objects")
