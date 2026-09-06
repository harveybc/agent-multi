"""PRE freeze for order C27-C28: the activation contradiction is
PHYSICAL — pinning the reviewed authorization digest into the
executor changes its file hash and the live amendment-10 chain
check refuses the very code needed to consume the authorization.
This probe edits the real file on disk (not memory) and restores
the exact bytes afterwards. Kept as a permanent regression."""
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
EXE = REPO / "tools/b4_campaign_executor.py"
A10_EXECUTOR_PIN = ("c852c3cc5dd8469507f1238d07db60cf77c72bf2596"
                    "6e843d4911e6af93c6cfc")
AUTH_SHA = ("c58008cc5285365b4c64e2827a9b9d1a329e3b64f7c72a37b62"
            "c1c6e702ae55d")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def chain_check() -> str:
    """Run the chain verifier in a FRESH process so the physical
    file identity boundary is exercised."""
    rc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'tools');\n"
         "import b4_authority as a\n"
         "try:\n"
         "    a.verify_amendment_chain(); print('CHAIN_OK')\n"
         "except SystemExit as e:\n"
         "    print('CHAIN_REFUSED:', str(e)[:90])"],
        cwd=REPO, capture_output=True, text=True)
    return (rc.stdout + rc.stderr).strip()


original = EXE.read_bytes()
print("1. amendment-10 executor pin:", A10_EXECUTOR_PIN[:16], "...")
print("   live executor file hash :", sha(original)[:16], "...")
assert sha(original) == A10_EXECUTOR_PIN
print("2. executor contains CAMPAIGN_AUTH_SHA = None:",
      b"CAMPAIGN_AUTH_SHA = None" in original)
assert b"CAMPAIGN_AUTH_SHA = None" in original
print("   baseline chain:", chain_check())
assert "CHAIN_OK" in chain_check()

activated = original.replace(
    b"CAMPAIGN_AUTH_SHA = None",
    b'CAMPAIGN_AUTH_SHA = "' + AUTH_SHA.encode() + b'"')
assert activated != original
try:
    EXE.write_bytes(activated)
    print("3. activated executor hash:", sha(activated)[:16],
          "... (differs from the pin)")
    out = chain_check()
    print("   chain with activation  :", out)
    assert "CHAIN_REFUSED" in out and "differs" in out
finally:
    EXE.write_bytes(original)
assert sha(EXE.read_bytes()) == A10_EXECUTOR_PIN
print("   bytes restored exactly :", True)
print("4. -> the owner decision is valid yet the campaign cannot "
      "be executed: consuming the authorization contradicts the "
      "append-only chain (finite two-phase activation required)")
print("\nPRE CONFIRMED: C27 activation contradiction reproduces "
      "physically")
