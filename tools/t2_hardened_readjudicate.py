#!/usr/bin/env python3
"""T2-R24/R25 (order 2026-09-12): a read-only readjudication run by the
HARDENED verifier, gated before a single evidence module is imported.

The historical execution keeps its commit and its bytes forever. This new
act runs the current hardened code — whose verify_unit_record refuses an
omitted mlp seed with a typed refusal instead of a KeyError — and binds
both identities in one external record,

    agent_multi.t2_hardened_readjudication_review_record.v1

at a fixed pathname under the reviewer-authority root.

This file is the ENTRY POINT and it uses only the standard library. Before
anything from the readjudication surface is imported it checks, in order:
no surface module already present in sys.modules; no PYTHONPATH; no import
path outside the checkout that could shadow a surface module; no bytecode
for a surface module; the record (strict JSON, exact schema, reviewer,
decision, date, scope, grants all false); the historical execution record
it binds; the clean checkout commit and tree; every surface file's digest;
and the preserved root's inventory of components and leaves. Only then
are modules imported — from the one checkout — and their bytes are hashed
again, so a file changed between the gate and the import refuses. Identity
is revalidated before and after the replay.

The candidate side can only build a template. It never writes a record.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import stat as _stat
import subprocess
import sys
from pathlib import Path

SCHEMA = "agent_multi.t2_hardened_readjudication_review_record.v1"
RECORD_NAME = "MUSASHI_T2_HARDENED_READJUDICATION_REVIEW_RECORD.json"
HISTORICAL_RECORD_NAME = "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json"
AUTHORITY_ROOT = Path.home() / ".config/agent-multi/reviewer_authority"
REVIEWER = "General Musashi"
DECISION = "OPEN_T2_READ_ONLY_HARDENED_READJUDICATION"
SCOPE = "READ_ONLY_HARDENED_READJUDICATION"
TEMPLATE_REVIEWER = "UNREVIEWED_TEMPLATE"
TEMPLATE_DECISION = "PENDING_EXTERNAL_REVIEW"
FIXTURE_REVIEWER = "ISOLATED_TEST_FIXTURE"
FIXTURE_DECISION = "FIXTURE_READ_ONLY_HARDENED_READJUDICATION"
FIXTURE_KIND = "ISOLATED_FIXTURE_NOT_EXTERNAL_REVIEW"

SURFACE = (
    "tools/t2_confirmatory.py", "tools/t2_confirmatory_executor.py",
    "tools/t2_assay_harness.py", "tools/t2_bank.py",
    "tools/t2_bank_census.py", "tools/t2_fresh_verifier.py",
    "tools/t2_public_data_census.py", "tools/t2_completion_reconstruction.py",
    "tools/t2_campaign_closure.py", "tools/descriptor_custody.py",
    "tools/t2_hardened_readjudicate.py",
)
SURFACE_MODULES = tuple(Path(p).stem for p in SURFACE)

KEYS = {
    "schema": str, "reviewer": str, "decision": str, "reviewed_at_date": str,
    "scope": str,
    "historical_execution_record_sha256": str, "historical_pinned_commit": str,
    "historical_pinned_tree": str, "historical_code_identity": dict,
    "hardened_commit": str, "hardened_tree": str, "hardened_surface": dict,
    "hardened_surface_sha256": str,
    "preserved_root_logical_id": str, "preserved_root_inventory_sha256": str,
    "candidate_adjudication_sha256": str,
    "retraining": bool, "downloads": bool, "model_execution": bool,
    "promotion": bool, "grants_execution": bool,
}
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")


class GateRefusal(SystemExit):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_obj(o) -> str:
    return sha_bytes(json.dumps(o, sort_keys=True,
                                separators=(",", ":")).encode())


def strict_json(raw: bytes, what: str):
    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise GateRefusal("RECORD_SCHEMA", f"{what}: duplicate keys")
        return dict(items)

    def constant(c):
        raise GateRefusal("RECORD_SCHEMA", f"{what}: non-finite constant {c}")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except ValueError as exc:
        raise GateRefusal("RECORD_SCHEMA", f"{what}: not JSON ({exc})")


def private_read(path: Path, missing_code: str) -> bytes:
    """Descriptor-first read of one private authority object: every
    component O_NOFOLLOW, the last two directories owned by this uid at
    exactly 0700, the file regular, same uid, exactly 0600."""
    parts = Path(path).parts
    if not os.path.lexists(path):
        raise GateRefusal(missing_code, f"{Path(path).name} does not exist at "
                                        "its fixed pathname")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i, comp in enumerate(parts[1:-1], start=1):
            nfd = os.open(comp, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY,
                          dir_fd=fd)
            os.close(fd)
            fd = nfd
            if len(parts) - 1 - i <= 2:
                st = os.fstat(fd)
                if st.st_uid != os.getuid() or _stat.S_IMODE(st.st_mode) != 0o700:
                    raise GateRefusal("RECORD_CUSTODY",
                                      f"authority directory {comp!r} is not "
                                      "owned 0700")
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    finally:
        os.close(fd)
    try:
        st = os.fstat(leaf)
        if not _stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or \
                _stat.S_IMODE(st.st_mode) != 0o600:
            raise GateRefusal("RECORD_CUSTODY", "authority record is not a "
                                                "private 0600 regular file")
        chunks = []
        while True:
            b = os.read(leaf, 1 << 20)
            if not b:
                break
            chunks.append(b)
        after = os.fstat(leaf)
        if (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != \
                (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns):
            raise GateRefusal("RECORD_CUSTODY", "authority record moved while read")
        return b"".join(chunks)
    finally:
        os.close(leaf)


def _git(co: Path, *args) -> str:
    r = subprocess.run(("git", "-C", str(co), *args), capture_output=True, text=True)
    if r.returncode:
        raise GateRefusal("CHECKOUT_MISMATCH", f"git {' '.join(args)} failed")
    return r.stdout.strip()


def checkout_facts(co: Path) -> dict:
    status = _git(co, "status", "--porcelain=v1", "--untracked-files=all")
    return {"commit": _git(co, "rev-parse", "HEAD"),
            "tree": _git(co, "rev-parse", "HEAD^{tree}"),
            "clean": status == "", "dirty": status.splitlines()[:20]}


def surface_digests(co: Path) -> dict:
    out, absent = {}, []
    for rel in SURFACE:
        p = Path(co) / rel
        if not p.is_file() or p.is_symlink():
            absent.append(rel)
            continue
        out[rel] = sha_bytes(p.read_bytes())
    if absent:
        raise GateRefusal("SURFACE_INCOMPLETE", f"missing {absent}")
    return out


def preserved_inventory(root: Path) -> dict:
    """Components AND leaves of the preserved root, from lstat facts."""
    root = Path(root)
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            p = Path(dirpath) / name
            st = os.lstat(p)
            entries.append([str(p.relative_to(root)), _stat.S_IFMT(st.st_mode),
                            st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns])
    st = os.lstat(root)
    return {"logical_id": root.name,
            "inventory_sha256": sha_obj({"root": [st.st_ino, st.st_mtime_ns,
                                                  st.st_ctime_ns],
                                         "entries": entries}),
            "components_and_leaves": len(entries)}


def environment_guard(co: Path) -> dict:
    loaded = sorted(m for m in SURFACE_MODULES if m in sys.modules)
    if loaded:
        raise GateRefusal("MODULE_PRELOADED",
                          f"surface modules already imported: {loaded}")
    if os.environ.get("PYTHONPATH"):
        raise GateRefusal("PYTHONPATH_SET",
                          "PYTHONPATH could shadow the reviewed checkout")
    tools = (Path(co) / "tools").resolve()
    for entry in sys.path:
        base = Path(entry or ".").resolve()
        if base == tools:
            continue
        for m in SURFACE_MODULES:
            if (base / f"{m}.py").exists() or (base / m).is_dir():
                raise GateRefusal("SHADOWED_IMPORT",
                                  f"{m} is importable from {base.name}, outside "
                                  "the reviewed checkout")
    cache = tools / "__pycache__"
    if cache.is_dir():
        stale = sorted(p.name for p in cache.iterdir()
                       if p.name.split(".")[0] in SURFACE_MODULES)
        if stale:
            raise GateRefusal("BYTECODE_PRESENT",
                              f"bytecode for surface modules exists: {stale[:4]}")
    for p in tools.glob("*.pyc"):
        raise GateRefusal("BYTECODE_PRESENT", f"bytecode without source: {p.name}")
    return {"sys_modules_clean": True, "pythonpath": None,
            "shadowing": "NONE", "bytecode": "NONE"}


def verify_record(co: Path, root: Path, *, authority_root: Path | None = None,
                  fixture: bool = False) -> dict:
    auth = Path(authority_root) if authority_root else AUTHORITY_ROOT
    real = AUTHORITY_ROOT.resolve() if AUTHORITY_ROOT.exists() else AUTHORITY_ROOT
    if fixture and (auth.resolve() == real or real in auth.resolve().parents):
        raise GateRefusal("FIXTURE_AT_AUTHORITY_ROOT",
                          "a fixture is never read from the reviewer-authority root")
    raw = private_read(auth / RECORD_NAME, "HARDENED_READJUDICATION_REVIEW_RECORD_REQUIRED")
    rec = strict_json(raw, "record")
    if not isinstance(rec, dict) or set(rec) != set(KEYS) or \
            any(type(rec[k]) is not t for k, t in KEYS.items()):
        raise GateRefusal("RECORD_SCHEMA", "not the exact v1 schema")
    if rec["schema"] != SCHEMA:
        raise GateRefusal("RECORD_SCHEMA", "foreign schema")
    want = (FIXTURE_REVIEWER, FIXTURE_DECISION) if fixture else (REVIEWER, DECISION)
    if (rec["reviewer"], rec["decision"]) != want:
        raise GateRefusal("RECORD_NOT_EXTERNALLY_REVIEWED" if not fixture
                          else "FIXTURE_NOT_MARKED_AS_FIXTURE",
                          "reviewer/decision not accepted in this mode")
    try:
        d = _dt.date.fromisoformat(rec["reviewed_at_date"])
    except ValueError:
        raise GateRefusal("RECORD_SCHEMA", "date not ISO")
    if d.isoformat() != rec["reviewed_at_date"]:
        raise GateRefusal("RECORD_SCHEMA", "date not canonical")
    if rec["scope"] != SCOPE or any(rec[k] for k in (
            "retraining", "downloads", "model_execution", "promotion",
            "grants_execution")):
        raise GateRefusal("SCOPE_OR_GRANT_VIOLATION",
                          "read-only hardened readjudication grants nothing")
    for k in ("historical_execution_record_sha256", "hardened_surface_sha256",
              "preserved_root_inventory_sha256", "candidate_adjudication_sha256"):
        if not HEX64.fullmatch(rec[k]):
            raise GateRefusal("RECORD_SCHEMA", f"{k} not hex64")
    for k in ("historical_pinned_commit", "historical_pinned_tree",
              "hardened_commit", "hardened_tree"):
        if not HEX40.fullmatch(rec[k]):
            raise GateRefusal("RECORD_SCHEMA", f"{k} not hex40")
    hist_raw = private_read(auth / HISTORICAL_RECORD_NAME,
                            "T2_SUCCESSOR_EXECUTION_RECORD_REQUIRED")
    hist = strict_json(hist_raw, "historical execution record")
    if sha_bytes(hist_raw) != rec["historical_execution_record_sha256"] or \
            hist.get("pinned_commit") != rec["historical_pinned_commit"] or \
            hist.get("pinned_tree") != rec["historical_pinned_tree"] or \
            hist.get("executor_code_identity") != rec["historical_code_identity"]:
        raise GateRefusal("HISTORICAL_RECORD_MISMATCH",
                          "the historical execution record is not the one reviewed")
    facts = checkout_facts(co)
    if not facts["clean"] or (facts["commit"], facts["tree"]) != (
            rec["hardened_commit"], rec["hardened_tree"]):
        raise GateRefusal("CHECKOUT_MISMATCH", "not the clean reviewed hardened checkout")
    surface = surface_digests(co)
    if surface != rec["hardened_surface"] or sha_obj(surface) != rec["hardened_surface_sha256"]:
        raise GateRefusal("SURFACE_MISMATCH", "hardened surface differs from the record")
    inv = preserved_inventory(root)
    if (inv["logical_id"], inv["inventory_sha256"]) != (
            rec["preserved_root_logical_id"], rec["preserved_root_inventory_sha256"]):
        raise GateRefusal("PRESERVED_ROOT_MISMATCH", "root differs from the record")
    return {"record_sha256": sha_bytes(raw),
            "record_kind": FIXTURE_KIND if fixture else "EXTERNAL_REVIEW_RECORD",
            "checkout": str(Path(co).resolve()), "facts": facts,
            "surface": surface, "surface_sha256": sha_obj(surface),
            "inventory": inv, "historical": {
                "record_sha256": rec["historical_execution_record_sha256"],
                "commit": rec["historical_pinned_commit"],
                "tree": rec["historical_pinned_tree"],
                "code_identity": dict(rec["historical_code_identity"])},
            "candidate_adjudication_sha256": rec["candidate_adjudication_sha256"],
            "scope": SCOPE}


def gate(co: Path, root: Path, *, authority_root=None, fixture=False) -> dict:
    env = environment_guard(co)
    verified = verify_record(co, root, authority_root=authority_root, fixture=fixture)
    verified["environment"] = env
    return verified


def import_surface(verified: dict):
    """Import from the one checkout, then prove the imported bytes are
    the bytes the gate hashed."""
    co = Path(verified["checkout"])
    tools = str(co / "tools")
    sys.dont_write_bytecode = True
    while tools in sys.path:
        sys.path.remove(tools)
    sys.path.insert(0, tools)
    import t2_campaign_closure as closure  # noqa: E402
    conf, ex, recon, dc = closure.load_hardened_modules(co)
    # every surface module that is now loaded — including those imported
    # indirectly by t2_confirmatory — must come from the checkout and
    # still carry the bytes the gate hashed
    for name in SURFACE_MODULES:
        mod = sys.modules.get(name)
        if mod is None or name == Path(__file__).stem:
            continue
        path = Path(mod.__file__).resolve()
        if path.parent != Path(tools).resolve():
            raise GateRefusal("IMPORT_MIX", f"{name} resolved outside the checkout")
        rel = str(path.relative_to(co))
        if sha_bytes(path.read_bytes()) != verified["surface"].get(rel):
            raise GateRefusal("IMPORT_IDENTITY_MOVED",
                              f"{rel} changed between the gate and the import")
    return closure, conf, ex, recon, dc


def revalidate(verified: dict, root: Path, stage: str) -> None:
    co = Path(verified["checkout"])
    facts = checkout_facts(co)
    if not facts["clean"] or facts["commit"] != verified["facts"]["commit"] or \
            surface_digests(co) != verified["surface"]:
        raise GateRefusal("CHECKOUT_MISMATCH", f"checkout moved {stage}")
    if preserved_inventory(root)["inventory_sha256"] != verified["inventory"]["inventory_sha256"]:
        raise GateRefusal("PRESERVED_ROOT_MISMATCH", f"root moved {stage}")


def historical_code_identity_view(verified: dict):
    """Claims and the historical record pin the HISTORICAL executor code;
    the verifier now runs hardened code. The code identity they are
    compared with is the historical one the record bound — after
    re-checking that the running surface is still the reviewed one."""
    def view(repo_root=None):
        if surface_digests(Path(verified["checkout"])) != verified["surface"]:
            raise GateRefusal("SURFACE_MISMATCH", "surface moved during replay")
        return dict(verified["historical"]["code_identity"])
    return view


def historical_head_view(verified: dict):
    def view():
        facts = checkout_facts(Path(verified["checkout"]))
        if not facts["clean"] or facts["commit"] != verified["facts"]["commit"]:
            raise GateRefusal("CHECKOUT_MISMATCH", "checkout moved during replay")
        return (verified["historical"]["commit"], verified["historical"]["tree"])
    return view


def checkout_gate(verified: dict):
    def gate_fn(pinned_commit, pinned_tree, repo_root=None):
        if (pinned_commit, pinned_tree) != (verified["historical"]["commit"],
                                            verified["historical"]["tree"]):
            raise GateRefusal("HISTORICAL_RECORD_MISMATCH", "consumed record differs")
        revalidate_checkout_only(verified)
    return gate_fn


def revalidate_checkout_only(verified: dict) -> None:
    co = Path(verified["checkout"])
    facts = checkout_facts(co)
    if not facts["clean"] or facts["commit"] != verified["facts"]["commit"]:
        raise GateRefusal("CHECKOUT_MISMATCH", "checkout moved")


def build_template(co: Path, root: Path, candidate_sha256: str,
                   authority_root: Path | None = None) -> dict:
    auth = Path(authority_root) if authority_root else AUTHORITY_ROOT
    hist_raw = private_read(auth / HISTORICAL_RECORD_NAME,
                            "T2_SUCCESSOR_EXECUTION_RECORD_REQUIRED")
    hist = strict_json(hist_raw, "historical execution record")
    facts = checkout_facts(co)
    surface = surface_digests(co)
    inv = preserved_inventory(root)
    return {"schema": SCHEMA, "reviewer": TEMPLATE_REVIEWER,
            "decision": TEMPLATE_DECISION, "reviewed_at_date": "YYYY-MM-DD",
            "scope": SCOPE,
            "historical_execution_record_sha256": sha_bytes(hist_raw),
            "historical_pinned_commit": hist["pinned_commit"],
            "historical_pinned_tree": hist["pinned_tree"],
            "historical_code_identity": dict(sorted(hist["executor_code_identity"].items())),
            "hardened_commit": facts["commit"], "hardened_tree": facts["tree"],
            "hardened_surface": surface, "hardened_surface_sha256": sha_obj(surface),
            "preserved_root_logical_id": inv["logical_id"],
            "preserved_root_inventory_sha256": inv["inventory_sha256"],
            "candidate_adjudication_sha256": candidate_sha256,
            "retraining": False, "downloads": False, "model_execution": False,
            "promotion": False, "grants_execution": False}


def write_template(dest: Path, template: dict, authority_root: Path | None = None) -> Path:
    auth = (Path(authority_root) if authority_root else AUTHORITY_ROOT).expanduser().resolve()
    dest = Path(dest).expanduser().resolve()
    if dest == auth or auth in dest.parents or template.get("reviewer") != TEMPLATE_REVIEWER:
        raise GateRefusal("CANDIDATE_MAY_NOT_WRITE_AUTHORITY",
                          "only an unreviewed template, never inside the authority root")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(template, indent=1, sort_keys=True) + "\n")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkout", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--template-out", type=Path)
    ap.add_argument("--candidate-evidence")
    ap.add_argument("--review-fixture-authority", type=Path)
    ap.add_argument("--template-in", type=Path)
    ap.add_argument("--submit", type=Path)
    ap.add_argument("--publication-commit")
    ap.add_argument("--divergence-out", type=Path,
                    help="where a CANDIDATE_ADJUDICATION_DIVERGES stop writes "
                         "its field-level report; no submission is written")
    a = ap.parse_args(argv)
    co = a.checkout.expanduser().resolve()
    root = a.root.expanduser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if a.template_out:
        # Template mode opens no evidence and authorizes nothing; it imports
        # the closure only to derive the corrected sign test for the candidate.
        environment_guard(co)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(co / "tools"))
        import t2_campaign_closure as closure  # noqa: E402
        ev = json.loads((co / a.candidate_evidence).read_text())
        cand = closure.scientific_adjudication_digest({
            "final_adjudication_counts": ev["final_adjudication_counts"],
            "screen_adjudication": ev["screen_adjudication"],
            "sign_test_supersession": closure.supersede_sign_test(ev["screen_adjudication"])})
        t = build_template(co, root, cand)
        write_template(a.template_out, t)
        print(json.dumps({"template_written": True, "evidence_opened": False,
                          "candidate_adjudication_sha256": cand}, indent=1))
        return 0
    fixture = a.review_fixture_authority is not None
    verified = gate(co, root, authority_root=a.review_fixture_authority, fixture=fixture)
    closure, conf, ex, recon, dc = import_surface(verified)
    out = closure.run_hardened(a, verified, sys.modules[__name__], conf, ex, recon, dc)
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
