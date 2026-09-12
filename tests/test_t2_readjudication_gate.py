"""T2-R18/R21: the readjudication gate refuses by its exact cause."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import descriptor_custody as DC  # noqa: E402
import t2_readjudication_gate as G  # noqa: E402

SEVEN = ("tools/t2_confirmatory.py", "tools/t2_confirmatory_executor.py",
         "tools/t2_assay_harness.py", "tools/t2_bank.py",
         "tools/t2_bank_census.py", "tools/t2_fresh_verifier.py",
         "tools/t2_public_data_census.py")


def git(repo, *args):
    subprocess.run(("git", "-C", str(repo), *args), check=True,
                   capture_output=True)


def make_checkout(tmp_path) -> Path:
    co = tmp_path / "reproducer"
    (co / "tools").mkdir(parents=True)
    for rel in G.READJUDICATION_SURFACE:
        (co / rel).write_text(f"# {rel}\n")
    git(co, "init", "-q")
    git(co, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(co, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    return co


def private_open(path, missing_msg=None):
    if not os.path.exists(path):
        raise SystemExit(missing_msg)
    st = os.stat(path)
    if st.st_mode & 0o077:
        raise SystemExit("authority file mode is not 0600")
    return os.open(path, os.O_RDONLY | os.O_NOFOLLOW)


def make_authority(tmp_path, co):
    auth = tmp_path / "authority"
    auth.mkdir(mode=0o700)
    hist = {"pinned_commit": "1" * 40, "pinned_tree": "2" * 40,
            "executor_code_identity": {
                rel: G.sha_bytes((co / rel).read_bytes()) for rel in SEVEN}}
    hp = auth / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json"
    hp.write_text(json.dumps(hist))
    os.chmod(hp, 0o600)
    conf = types.SimpleNamespace(
        AUTHORITY_ROOT=auth,
        T2_SUCCESSOR_EXECUTION_RECORD_PATH=hp,
        _open_private_authority_file=private_open)
    return auth, conf, hp


def make_root(tmp_path):
    root = tmp_path / "t2_root"
    (root / "units").mkdir(parents=True)
    (root / "units" / "RECORD_u.json").write_text("{}")
    (root / "units" / "ARRAYS_u.npz").write_bytes(b"x" * 32)
    return root


def install(auth, record: dict):
    p = auth / G.RECORD_NAME
    p.write_text(json.dumps(record))
    os.chmod(p, 0o600)
    return p


def reviewed(template: dict, **over) -> dict:
    rec = dict(template, reviewer=G.REVIEWER, decision=G.DECISION,
               reviewed_at_date="2026-09-13")
    rec.update(over)
    return rec


@pytest.fixture()
def world(tmp_path):
    co = make_checkout(tmp_path)
    auth, conf, hp = make_authority(tmp_path, co)
    root = make_root(tmp_path)
    preserved = G.preserved_root_identity(DC, root)
    template = G.build_template(historical_record_raw=hp.read_bytes(),
                                checkout=co, preserved=preserved,
                                candidate_adjudication_sha256="a" * 64)
    return types.SimpleNamespace(co=co, auth=auth, conf=conf, hp=hp,
                                 root=root, preserved=preserved,
                                 template=template, tmp=tmp_path)


def verify(w):
    return G.verify_record(w.conf, checkout=w.co, preserved=w.preserved,
                           authority_root=w.auth)


def code(exc):
    return exc.value.code


def test_without_a_record_it_stops_before_opening_evidence(world):
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.require_record(world.conf, authority_root=world.auth)
    assert code(e) == "READJUDICATION_REVIEW_RECORD_REQUIRED"


def test_a_reviewed_record_verifies(world):
    install(world.auth, reviewed(world.template))
    out = verify(world)
    assert out["scope"] == G.SCOPE
    G.assert_candidate_matches(out, "a" * 64)


def test_a_self_emitted_template_is_not_a_review(world):
    install(world.auth, world.template)
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "RECORD_NOT_EXTERNALLY_REVIEWED"


def test_the_candidate_cannot_write_into_the_authority_root(world):
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.write_template(world.auth / G.RECORD_NAME, world.template,
                         world.auth)
    assert code(e) == "CANDIDATE_MAY_NOT_WRITE_AUTHORITY"
    with pytest.raises(G.ReadjudicationRefusal):
        G.write_template(world.tmp / "out.json",
                         reviewed(world.template), world.auth)
    assert G.write_template(world.tmp / "out.json", world.template,
                            world.auth).is_file()


def test_a_retroactively_repinned_execution_record_refuses(world):
    install(world.auth, reviewed(world.template))
    hist = json.loads(world.hp.read_text())
    hist["pinned_commit"] = "9" * 40
    world.hp.write_text(json.dumps(hist))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "HISTORICAL_RECORD_MISMATCH"


def test_historical_bytes_absent_from_the_reproducer_refuse(world):
    install(world.auth, reviewed(world.template))
    (world.co / "tools/t2_bank.py").write_text("# changed\n")
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "HISTORICAL_CODE_MISMATCH"


def test_a_snapshot_with_an_omitted_file_refuses(world):
    (world.co / "tools/t2_completion_reconstruction.py").unlink()
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.readjudication_surface(world.co)
    assert code(e) == "SURFACE_INCOMPLETE"


def test_a_dirty_or_other_commit_reproducer_refuses(world):
    install(world.auth, reviewed(world.template))
    (world.co / "tools/stray.py").write_text("x = 1\n")
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "REPRODUCER_CHECKOUT_MISMATCH"
    (world.co / "tools/stray.py").unlink()
    install(world.auth, reviewed(world.template, reproducer_commit="3" * 40))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "REPRODUCER_CHECKOUT_MISMATCH"


def test_a_changed_surface_file_refuses(world):
    install(world.auth, reviewed(world.template,
                                 readjudication_surface=dict(
                                     world.template["readjudication_surface"],
                                     **{"tools/t2_campaign_closure.py":
                                        "0" * 64})))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "SURFACE_MISMATCH"


def test_a_different_preserved_root_refuses(world):
    install(world.auth, reviewed(world.template))
    (world.root / "units" / "ARRAYS_u.npz").write_bytes(b"y" * 33)
    world.preserved = G.preserved_root_identity(DC, world.root)
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "PRESERVED_ROOT_MISMATCH"


@pytest.mark.parametrize("over", [
    {"retraining": True}, {"downloads": True}, {"grants_execution": True},
    {"scope": "EXECUTION"}])
def test_a_record_that_grants_more_than_read_only_refuses(world, over):
    install(world.auth, reviewed(world.template, **over))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "SCOPE_OR_GRANT_VIOLATION"


@pytest.mark.parametrize("mut", [
    lambda r: r.update(extra=1), lambda r: r.pop("downloads"),
    lambda r: r.update(retraining=0),
    lambda r: r.update(reviewed_at_date="13/09/2026")])
def test_a_malformed_record_refuses(world, mut):
    rec = reviewed(world.template)
    mut(rec)
    install(world.auth, rec)
    with pytest.raises(G.ReadjudicationRefusal) as e:
        verify(world)
    assert code(e) == "RECORD_SCHEMA"


def test_a_diverging_recomputed_adjudication_stops(world):
    install(world.auth, reviewed(world.template))
    out = verify(world)
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.assert_candidate_matches(out, "b" * 64)
    assert code(e) == "CANDIDATE_ADJUDICATION_DIVERGES"


def test_the_gate_never_writes_into_the_authority_root():
    import ast
    src = (REPO / "tools/t2_readjudication_gate.py").read_text()
    writes = [n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.Attribute) and n.attr in (
                  "write_text", "write_bytes")]
    assert len(writes) == 1, "only write_template writes, and it refuses " \
                             "the authority root"


# ------------------------------------------------ R20 fixture isolation
def fixture_rec(template, **over):
    rec = dict(template, reviewer=G.FIXTURE_REVIEWER,
               decision=G.FIXTURE_DECISION, reviewed_at_date="2026-09-13")
    rec.update(over)
    return rec


def test_an_isolated_fixture_verifies_only_in_fixture_mode(world, tmp_path):
    fx = tmp_path / "fixture_authority"
    fx.mkdir(mode=0o700)
    shutil.copy(world.hp, fx / world.hp.name)
    os.chmod(fx / world.hp.name, 0o600)
    install(fx, fixture_rec(world.template))
    out = G.verify_record(world.conf, checkout=world.co,
                          preserved=world.preserved, authority_root=fx,
                          fixture=True)
    assert out["record_kind"] == "ISOLATED_FIXTURE_NOT_AN_EXTERNAL_RECORD"
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.verify_record(world.conf, checkout=world.co,
                        preserved=world.preserved, authority_root=fx)
    assert code(e) == "RECORD_NOT_EXTERNALLY_REVIEWED"


def test_a_fixture_at_the_real_authority_root_refuses(world):
    install(world.auth, fixture_rec(world.template))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.verify_record(world.conf, checkout=world.co,
                        preserved=world.preserved,
                        authority_root=world.auth, fixture=True)
    assert code(e) == "FIXTURE_AT_AUTHORITY_ROOT"


def test_a_fixture_carrying_the_reviewers_name_refuses(world, tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir(mode=0o700)
    shutil.copy(world.hp, fx / world.hp.name)
    os.chmod(fx / world.hp.name, 0o600)
    install(fx, reviewed(world.template))
    with pytest.raises(G.ReadjudicationRefusal) as e:
        G.verify_record(world.conf, checkout=world.co,
                        preserved=world.preserved, authority_root=fx,
                        fixture=True)
    assert code(e) == "FIXTURE_NOT_MARKED_AS_FIXTURE"


def test_the_scientific_digest_ignores_identity_and_time_but_not_effects():
    base = {"final_adjudication_counts": {"COMPLETED_VERIFIED": 242},
            "screen_adjudication": {"verdict": "DOES_NOT_ADVANCE",
                                    "primary": -0.001048},
            "sign_test_supersession": {"corrected_value": 1.0,
                                       "corrected_table_0_to_6": [1],
                                       "signs_positive": 3},
            "inventory": {"exact": True, "sealed_units": 242,
                          "total_artifacts": 726},
            "closed_at": "a", "reviewed_identity": {"x": 1}}
    other = dict(base, closed_at="b", reviewed_identity={"x": 2})
    assert G.scientific_adjudication_digest(base) == \
        G.scientific_adjudication_digest(other)
    moved = json.loads(json.dumps(base))
    moved["screen_adjudication"]["primary"] = -0.001049
    assert G.scientific_adjudication_digest(base) != \
        G.scientific_adjudication_digest(moved)


# ---------------------------------------------------- R21 import mixing
def test_a_closure_run_from_another_tree_is_an_import_mix(tmp_path):
    import t2_campaign_closure as T
    other = tmp_path / "elsewhere"
    (other / "tools").mkdir(parents=True)
    with pytest.raises(SystemExit) as e:
        T.load_single_checkout_modules(other)
    assert "IMPORT_MIX" in str(e.value)


def test_the_scientific_digest_does_not_depend_on_the_inventory_block():
    base = {"final_adjudication_counts": {"COMPLETED_VERIFIED": 242},
            "screen_adjudication": {"verdict": "DOES_NOT_ADVANCE"},
            "sign_test_supersession": {"corrected_value": 1.0,
                                       "corrected_table_0_to_6": [1],
                                       "signs_positive": 3}}
    with_inv = dict(base, inventory={"exact": True, "total_artifacts": 726})
    assert G.scientific_adjudication_digest(base) == \
        G.scientific_adjudication_digest(with_inv)


# ------------------------------------------ R19 scoped checkout gate
def _verified(world):
    install(world.auth, reviewed(world.template))
    return verify(world)


def test_the_readjudication_checkout_gate_accepts_the_reviewed_reproducer(world):
    v = _verified(world)
    gate = G.readjudication_checkout_gate(v)
    gate(v["historical_pinned_commit"], v["historical_pinned_tree"], world.co)


def test_the_readjudication_checkout_gate_refuses_another_historical_pin(world):
    v = _verified(world)
    gate = G.readjudication_checkout_gate(v)
    with pytest.raises(G.ReadjudicationRefusal) as e:
        gate("9" * 40, v["historical_pinned_tree"], world.co)
    assert code(e) == "HISTORICAL_RECORD_MISMATCH"


def test_the_readjudication_checkout_gate_refuses_a_dirty_reproducer(world):
    v = _verified(world)
    gate = G.readjudication_checkout_gate(v)
    (world.co / "tools" / "stray.py").write_text("x = 1\n")
    with pytest.raises(G.ReadjudicationRefusal) as e:
        gate(v["historical_pinned_commit"], v["historical_pinned_tree"], world.co)
    assert code(e) == "REPRODUCER_CHECKOUT_MISMATCH"


def test_the_scoped_identity_restores_everything_even_on_exception():
    import t2_campaign_closure as T
    conf = types.SimpleNamespace(verify_executor_checkout="orig_gate")
    ex = types.SimpleNamespace(_git_head_tree="orig_head", main="orig_main",
                               rehearse="orig_rehearse")
    repl = {("conf", "verify_executor_checkout"): "g",
            ("ex", "_git_head_tree"): "h",
            ("ex", "main"): T._write_path_refusal("main"),
            ("ex", "rehearse"): T._write_path_refusal("rehearse")}
    with pytest.raises(RuntimeError):
        with T.scoped_readjudication_identity(conf, ex, repl):
            assert ex._git_head_tree == "h"
            with pytest.raises(SystemExit) as e:
                ex.main()
            assert "READ_ONLY_READJUDICATION" in str(e.value)
            with pytest.raises(SystemExit):
                ex.rehearse()
            raise RuntimeError("boom")
    assert (conf.verify_executor_checkout, ex._git_head_tree, ex.main,
            ex.rehearse) == ("orig_gate", "orig_head", "orig_main",
                             "orig_rehearse")


def test_the_historical_identity_view_returns_the_reviewed_pins(world):
    v = _verified(world)
    view = G.historical_identity_view(v, world.co)
    assert view() == (v["historical_pinned_commit"],
                      v["historical_pinned_tree"])


def test_the_historical_identity_view_refuses_a_moved_reproducer(world):
    v = _verified(world)
    view = G.historical_identity_view(v, world.co)
    (world.co / "tools" / "stray.py").write_text("x = 1\n")
    with pytest.raises(G.ReadjudicationRefusal) as e:
        view()
    assert code(e) == "REPRODUCER_CHECKOUT_MISMATCH"


def test_the_legacy_modes_never_install_the_replacement():
    import ast
    src = (REPO / "tools/t2_campaign_closure.py").read_text()
    tree = ast.parse(src)
    users = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
             and "scoped_readjudication_identity(" in ast.get_source_segment(src, n)]
    assert users == ["run_reproducer"], users


# ------------------------------------------------ R20 submission v3
def _closure_stub(kind="ISOLATED_FIXTURE_NOT_AN_EXTERNAL_RECORD", equal=True):
    screen = {"verdict": "DOES_NOT_ADVANCE",
              "primary_estimand_unweighted_mean_of_panel_effects": -0.001048,
              "panels": {"a": {"effect": 0.1}, "b": {"effect": -0.2}}}
    return {
        "campaign_root_logical": "t2_successor",
        "screen_adjudication": screen,
        "final_adjudication_counts": {"COMPLETED_VERIFIED": 242,
                                      "TERMINAL_FAILED": 0},
        "sign_test_supersession": {"corrected_value": 1.0},
        "inventory": {"exact": True},
        "measurement": {"custody_reads": 726,
                        "leaf_binding": {"reads_total": 726,
                                         "reads_bound": 726}},
        "readjudication_identity": {
            "historical_pinned_commit": "1" * 40,
            "historical_pinned_tree": "2" * 40,
            "historical_execution_record_sha256": "3" * 64,
            "reproducer": {"commit": "4" * 40, "tree": "5" * 40},
            "surface_sha256": "6" * 64, "record_kind": kind,
            "record_sha256": "7" * 64,
            "candidate_adjudication_sha256": "8" * 64,
            "scientific_adjudication_sha256": "8" * 64 if equal else "9" * 64,
            "executor_checkout_gate": "REPLACED...",
            "executor_claim_identity": "REPLACED..."},
    }


def test_the_v3_submission_separates_historical_from_readjudication():
    import t2_campaign_closure as T
    c = _closure_stub()
    sub = T.build_readjudication_submission_v3(
        c, template={"x": 1}, evidence_rel="docs/e.json",
        evidence_sha256="e" * 64,
        historical_screen=c["screen_adjudication"],
        publication_commit="a" * 40)
    assert sub["schema"].endswith(".v3")
    assert sub["historical_result"]["identity"].endswith("2" * 40)
    assert sub["readjudication"]["identity"].count("4" * 40) == 1
    assert sub["readjudication"]["equal_to_candidate"] is True
    assert all(sub["readjudication"]["panel_effects_equal_to_historical"].values())
    assert sub["requires"] == "EXTERNAL_READJUDICATION_REVIEW_RECORD"
    assert sub["grants_promotion"] is False
    blob = json.dumps(sub)
    assert "/home/" not in blob and "ISOLATED_FIXTURE" in blob


def test_the_v3_submission_digest_covers_the_publication_commit():
    import t2_campaign_closure as T
    c = _closure_stub()
    kw = dict(template={}, evidence_rel="e", evidence_sha256="e" * 64,
              historical_screen=c["screen_adjudication"])
    a = T.build_readjudication_submission_v3(c, publication_commit="a" * 40, **kw)
    b = T.build_readjudication_submission_v3(c, publication_commit="b" * 40, **kw)
    assert a["publication"]["commit_a"] != b["publication"]["commit_a"]
    assert a["submission_sha256"] != b["submission_sha256"]
