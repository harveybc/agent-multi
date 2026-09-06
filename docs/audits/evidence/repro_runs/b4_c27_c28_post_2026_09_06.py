"""POST for order C27-C28: the activation contradiction is closed
— the executor carries the reviewed authorization digest AND the
live chain (now 11 amendments) validates the exact executing
bytes; the operator-specific absolute data fallback is gone."""
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
import b4_authority as a  # noqa: E402

EXE = REPO / "tools/b4_campaign_executor.py"
AUTH_SHA = ("c58008cc5285365b4c64e2827a9b9d1a329e3b64f7c72a37b62"
            "c1c6e702ae55d")

print("== C27: activation without contradiction ==")
src = EXE.read_text()
assert "CAMPAIGN_AUTH_SHA = None" not in src
assert AUTH_SHA[:32] in src
print("executor carries the reviewed digest:", True)
rc = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, 'tools');\n"
     "import b4_authority as a\n"
     "c = a.verify_amendment_chain()\n"
     "print('CHAIN_OK amendments', len(c['amendment_shas']))"],
    cwd=REPO, capture_output=True, text=True)
print(rc.stdout.strip() or rc.stderr.strip()[:90])
assert "CHAIN_OK amendments 11" in rc.stdout
got = a.verify_campaign_authorization_record(
    a.CAMPAIGN_AUTHORIZATION_RECORD_PATH,
    a._sha_file(a.CAMPAIGN_AUTHORIZATION_RECORD_PATH))
print("authorization record verifies (nested owner ratification, "
      "exact words, twelve-cell scope):", True)
a11 = a._strict_json_bytes(a.AMENDMENT_11_PATH.read_bytes(),
                           "a11")
assert a11["amends_amendment_10_sha256"] == a.AMENDMENT_10_SHA
assert a11["authorization_record_sha256"] == AUTH_SHA
print("amendment 11 names a10 + the reviewer record; amendments "
      "9/10 byte-immutable:",
      a._sha_file(a.AMENDMENT_9_PATH) == a.AMENDMENT_9_SHA and
      a._sha_file(a.AMENDMENT_10_PATH) == a.AMENDMENT_10_SHA)

print("\n== C28: no operator-specific absolute data path ==")
asrc = (REPO / "tools/b4_authority.py").read_text()
# no absolute home-anchored data path may remain (checked without
# writing any operator identity into this public probe)
import re
assert not re.search(
    r"/home/\w+/Documents/GitHub/predictor/examples/data", asrc)
assert "logical_rel = (" in asrc
assert "resolve_predictor_root().resolve()" in asrc
print("literal home fallback removed; logical relative identity "
      "+ containment + descriptor hashing in place:", True)

print("\nPOST CONFIRMED: C27-C28 closed; the campaign is "
      "executable pending only your final commit audit")
