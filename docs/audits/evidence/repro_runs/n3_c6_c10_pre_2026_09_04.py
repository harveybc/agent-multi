"""PRE freeze for order @a1e7b739 (C6-C10): the eight semantic
mutations P1-P8, each starting from the published v2 bundle, changing
ONE fact, recomputing the unit self-digest when needed, and supplying
the CANDIDATE'S OWN sha256 to cross the byte layer — all eight must
currently return N3_BUNDLE_VERIFIED, proving the 'exact schemas at
every level' claim false and the authority conflation real."""
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "n3f", REPO / "tools" / "n3_fresh_confirmation.py")
n3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3f)
from agent_plugins.experiment_runtime import sha_obj  # noqa: E402

V2 = (REPO / "docs/audits/evidence/"
      "N3_FRESH_CONFIRMATION_BUNDLE_V2_2026_09_04.json")
orig = json.loads(V2.read_text())


def redigest(u):
    u["payload_sha256"] = sha_obj(
        {k: v for k, v in u.items() if k != "payload_sha256"})


def probe(tag, mutate):
    b = copy.deepcopy(orig)
    mutate(b)
    p = Path.home() / ".cache" / f"c6_pre_{tag}.json"
    p.write_text(json.dumps(b, default=float))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        out = n3f.verify(p, sha)
        print(f"{tag}: ACCEPTED -> {out['verdict']}")
        return True
    except n3f.FreshRefusal as r:
        print(f"{tag}: refused ({str(r)[:70]})")
        return False
    finally:
        p.unlink(missing_ok=True)


def p1(b):
    b["decision_constants"]["margin_repr"] = 999


def p2(b):
    b["role_ledger"]["stride"] = 999


def p3(b):
    u = [x for x in b["units"] if x["horizon"] == 6][0]
    u["horizon"] = 999
    redigest(u)


def p4(b):
    b["contract"] = "docs/README_UNRELATED.md"


def p5(b):
    b["digests"]["acquired_parquet"] = "0" * 64


def p6(b):
    # support-preserving boolean: replace a label that IS 1 with
    # True (True == 1 keeps every derived count identical)
    u = b["units"][0]
    idx = u["labels"].index(1)
    u["labels"][idx] = True
    redigest(u)


def p7(b):
    u = b["units"][1]
    u["arms"]["arm2"]["probs"][0][0] = str(
        u["arms"]["arm2"]["probs"][0][0])
    redigest(u)


def p8(b):
    b["digests"]["undeclared_extra"] = "f" * 64


accepted = [probe(f"P{i}", fn) for i, fn in
            enumerate((p1, p2, p3, p4, p5, p6, p7, p8), 1)]
print(f"\nPRE CONFIRMED: {sum(accepted)}/8 semantic mutations "
      "accepted by the current verifier when handed their own "
      "digest — 'exact schemas at every level' is FALSE and a "
      "self-chosen checksum is being treated as authority")
assert all(accepted)
