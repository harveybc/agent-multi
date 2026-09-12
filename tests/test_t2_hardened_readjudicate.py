"""T2-R24..R27 battery for the hardened readjudicator."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_hardened_readjudicate as H  # noqa: E402

SEVEN = ("tools/t2_confirmatory.py", "tools/t2_confirmatory_executor.py",
         "tools/t2_assay_harness.py", "tools/t2_bank.py", "tools/t2_bank_census.py",
         "tools/t2_fresh_verifier.py", "tools/t2_public_data_census.py")


def git(repo, *args):
    subprocess.run(("git", "-C", str(repo), *args), check=True, capture_output=True)


@pytest.fixture()
def w(tmp_path, monkeypatch):
    co = tmp_path / "hardened"
    (co / "tools").mkdir(parents=True)
    for rel in H.SURFACE:
        (co / rel).write_text(f"# {rel}\n")
    git(co, "init", "-q")
    git(co, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(co, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    root = tmp_path / "t2_root"
    (root / "units").mkdir(parents=True)
    (root / "units" / "RECORD_u.json").write_text("{}")
    auth = tmp_path / "authority"
    auth.mkdir(mode=0o700)
    os.chmod(tmp_path, 0o700)
    hist = {"pinned_commit": "1" * 40, "pinned_tree": "2" * 40,
            "executor_code_identity": {r: "3" * 64 for r in SEVEN}}
    hp = auth / H.HISTORICAL_RECORD_NAME
    hp.write_text(json.dumps(hist))
    os.chmod(hp, 0o600)
    monkeypatch.setattr(H, "AUTHORITY_ROOT", auth)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    for m in H.SURFACE_MODULES:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != (REPO / "tools").resolve()])
    template = H.build_template(co, root, "a" * 64, authority_root=auth)
    return type("W", (), dict(co=co, root=root, auth=auth, template=template, tmp=tmp_path))


def install(auth, rec):
    p = auth / H.RECORD_NAME
    p.write_text(json.dumps(rec))
    os.chmod(p, 0o600)


def reviewed(t, **over):
    r = dict(t, reviewer=H.REVIEWER, decision=H.DECISION, reviewed_at_date="2026-09-13")
    r.update(over)
    return r


def fixture_rec(t, **over):
    r = dict(t, reviewer=H.FIXTURE_REVIEWER, decision=H.FIXTURE_DECISION,
             reviewed_at_date="2026-09-13")
    r.update(over)
    return r


def code(exc):
    return exc.value.code


def test_missing_record_refuses_before_any_surface_import(w):
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "HARDENED_READJUDICATION_REVIEW_RECORD_REQUIRED"
    assert not any(m in sys.modules for m in H.SURFACE_MODULES if m != "t2_hardened_readjudicate")


def test_a_reviewed_record_passes_the_gate(w):
    install(w.auth, reviewed(w.template))
    v = H.gate(w.co, w.root)
    assert v["record_kind"] == "EXTERNAL_REVIEW_RECORD"


def test_a_self_emitted_template_refuses(w):
    install(w.auth, w.template)
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "RECORD_NOT_EXTERNALLY_REVIEWED"


def test_the_candidate_cannot_write_into_the_authority_root(w):
    with pytest.raises(H.GateRefusal) as e:
        H.write_template(w.auth / "x.json", w.template, authority_root=w.auth)
    assert code(e) == "CANDIDATE_MAY_NOT_WRITE_AUTHORITY"


def test_an_isolated_fixture_is_accepted_only_outside_the_authority_root(w):
    fx = w.tmp / "fx"
    fx.mkdir(mode=0o700)
    shutil.copy(w.auth / H.HISTORICAL_RECORD_NAME, fx / H.HISTORICAL_RECORD_NAME)
    os.chmod(fx / H.HISTORICAL_RECORD_NAME, 0o600)
    install(fx, fixture_rec(w.template))
    v = H.gate(w.co, w.root, authority_root=fx, fixture=True)
    assert v["record_kind"] == H.FIXTURE_KIND == "ISOLATED_FIXTURE_NOT_EXTERNAL_REVIEW"
    install(w.auth, fixture_rec(w.template))
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root, authority_root=w.auth, fixture=True)
    assert code(e) == "FIXTURE_AT_AUTHORITY_ROOT"


def test_a_repinned_historical_record_refuses(w):
    install(w.auth, reviewed(w.template))
    hp = w.auth / H.HISTORICAL_RECORD_NAME
    h = json.loads(hp.read_text()); h["pinned_commit"] = "9" * 40
    hp.write_text(json.dumps(h))
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "HISTORICAL_RECORD_MISMATCH"


def test_a_changed_surface_file_refuses(w):
    install(w.auth, reviewed(w.template))
    (w.co / "tools/t2_bank.py").write_text("# changed\n")
    git(w.co, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "y")
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) in ("CHECKOUT_MISMATCH", "SURFACE_MISMATCH")


def test_a_changed_preserved_root_refuses(w):
    install(w.auth, reviewed(w.template))
    (w.root / "units" / "late.json").write_text("{}")
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "PRESERVED_ROOT_MISMATCH"


@pytest.mark.parametrize("over", [{"retraining": True}, {"downloads": True},
                                  {"model_execution": True}, {"promotion": True},
                                  {"scope": "EXECUTION"}])
def test_a_record_granting_more_than_read_only_refuses(w, over):
    install(w.auth, reviewed(w.template, **over))
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "SCOPE_OR_GRANT_VIOLATION"


def test_a_record_with_a_duplicate_key_refuses(w):
    p = w.auth / H.RECORD_NAME
    raw = json.dumps(reviewed(w.template))
    p.write_text(raw[:-1] + ', "downloads": false}')
    os.chmod(p, 0o600)
    with pytest.raises(H.GateRefusal) as e:
        H.gate(w.co, w.root)
    assert code(e) == "RECORD_SCHEMA"


def test_a_preloaded_surface_module_refuses(w, monkeypatch):
    monkeypatch.setitem(sys.modules, "t2_campaign_closure", object())
    with pytest.raises(H.GateRefusal) as e:
        H.environment_guard(w.co)
    assert code(e) == "MODULE_PRELOADED"


def test_pythonpath_refuses(w, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    with pytest.raises(H.GateRefusal) as e:
        H.environment_guard(w.co)
    assert code(e) == "PYTHONPATH_SET"


def test_a_shadowing_path_entry_refuses(w, monkeypatch):
    shadow = w.tmp / "shadow"
    shadow.mkdir()
    (shadow / "t2_bank.py").write_text("x = 1\n")
    monkeypatch.setattr(sys, "path", [str(shadow)] + sys.path)
    with pytest.raises(H.GateRefusal) as e:
        H.environment_guard(w.co)
    assert code(e) == "SHADOWED_IMPORT"


def test_bytecode_for_a_surface_module_refuses(w):
    cache = w.co / "tools" / "__pycache__"
    cache.mkdir()
    (cache / "t2_bank.cpython-312.pyc").write_bytes(b"\x00")
    with pytest.raises(H.GateRefusal) as e:
        H.environment_guard(w.co)
    assert code(e) == "BYTECODE_PRESENT"


def test_a_checkout_mutated_between_gate_and_import_refuses(w, monkeypatch):
    """The gate hashes the surface; if a file changes before import, the
    imported bytes no longer match."""
    install(w.auth, reviewed(w.template))
    v = H.gate(w.co, w.root)
    real = REPO / "tools"
    for rel in H.SURFACE:
        shutil.copy(real / Path(rel).name, w.co / rel)
    v["checkout"] = str(w.co.resolve())
    with pytest.raises((H.GateRefusal, SystemExit)) as e:
        H.import_surface(v)
    for m in H.SURFACE_MODULES:
        sys.modules.pop(m, None)
    assert "IMPORT_IDENTITY_MOVED" in str(e.value) or "IMPORT_MIX" in str(e.value)


def test_revalidation_refuses_a_moved_checkout(w):
    install(w.auth, reviewed(w.template))
    v = H.gate(w.co, w.root)
    (w.co / "tools" / "stray.py").write_text("x = 1\n")
    with pytest.raises(H.GateRefusal) as e:
        H.revalidate(v, w.root, "after replay")
    assert code(e) == "CHECKOUT_MISMATCH"


def test_the_historical_code_identity_view_refuses_a_moved_surface(w):
    install(w.auth, reviewed(w.template))
    v = H.gate(w.co, w.root)
    view = H.historical_code_identity_view(v)
    assert view() == v["historical"]["code_identity"]
    (w.co / "tools/t2_bank.py").write_text("# moved\n")
    with pytest.raises(H.GateRefusal):
        view()


def test_the_entry_point_imports_only_the_standard_library():
    import ast
    tree = ast.parse((REPO / "tools/t2_hardened_readjudicate.py").read_text())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "argparse", "datetime", "hashlib", "json", "os",
                   "re", "stat", "subprocess", "sys", "pathlib"}, top


def test_the_hardened_executor_types_an_omitted_seed():
    src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    assert "an omitted " in src and "seed never verifies" in src
