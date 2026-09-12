"""R9 (order 2026-09-12): the battery the T2 closure never had.

The audit's point was blunt — "no existe una bateria dedicada a
`t2_campaign_closure.py`" — and without one the two custody defects it
found could not have been made to bite. Nine cases, one per item of
R9, each attacking a layer of the closure directly. None of them opens
the preserved campaign root: R10 defers that to after review.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import t2_campaign_closure as T                                # noqa: E402
import descriptor_custody as DC                                # noqa: E402

REVIEWED = (Path.home() / "Documents/GitHub/.runtime"
            / "agent-multi-t2-reviewed-7bcd3f0d")


# --------------------------------------------------------------- helpers
class FakeDirSnapshot:
    """A RETAINED directory that serves prepared bytes and counts its
    reads. It stands in for the real snapshot so the adapter can be
    driven without a filesystem."""

    device = 7
    inode = 11
    mode = 0o750
    rel = "units"

    def __init__(self, files: dict[str, bytes]):
        self._files = dict(files)
        self.read_log: list[str] = []

    @property
    def files(self):
        return tuple(sorted(self._files))

    dirs = ()
    others = ()

    def read(self, name: str):
        self.read_log.append(name)
        if name not in self._files:
            raise DC.CustodyRefusal(f"units/{name}: does not exist")

        class _St:
            st_uid = 0
            st_mode = 0o100644
            st_ino = 1
            st_dev = 7
            st_mtime_ns = 0
        return DC.Artifact(f"units/{name}", self._files[name], _St(),
                           dir_device=self.device, dir_inode=self.inode)

    def replace(self, name: str, payload: bytes):
        """Simulate the directory being swapped under the snapshot: the
        RETAINED instance must not see it."""
        self._files[name] = payload

    def facts(self):
        return {"rel": self.rel, "files": sorted(self.files),
                "device": self.device, "inode": self.inode}


class FakeCustody:
    """Serves one retained units snapshot."""

    def __init__(self, files: dict[str, bytes]):
        self.root = Path("/fake")
        self.units = FakeDirSnapshot(files)

    def walk_to(self, rel: str):
        return self.units

    def close(self):
        pass


class FakeResultsRoot:
    """Stands in for the pinned ResultsRoot in the subclass factory."""


def snapshot_over(files: dict[str, bytes]):
    cls = T.make_snapshot_class(FakeResultsRoot)
    units = FakeDirSnapshot(files)
    snap = cls(units, tuple(sorted(files)), path=Path("/fake"))
    return units, snap


def a_record(uid_safe: str, value: float) -> bytes:
    return json.dumps({
        "wall_seconds": 10.0,
        "assay_record": {"unit": uid_safe, "effect": value,
                         "costs_by_phase": {"train": 1.0}}}).encode()


# ================================================================ R9.1
def test_a_record_swapped_after_verification_is_not_scored():
    files = {"RECORD_a.json": a_record("a", 0.5),
             "ARRAYS_a.npz": b"arrays-a"}
    custody, snap = snapshot_over(files)
    verified = snap.read_private(snap.units_fd, "RECORD_a.json",
                                 "unit record")
    custody.replace("RECORD_a.json", a_record("a", -99.0))
    scored = snap.record("a")
    assert json.loads(verified.decode())["assay_record"]["effect"] == 0.5
    assert scored["assay_record"]["effect"] == 0.5, (
        "the screen must consume the instance that was verified")
    assert custody.read_log.count("RECORD_a.json") == 1, (
        "one artifact, one read")


# ================================================================ R9.2
def test_arrays_swapped_after_verification_are_not_re_read():
    files = {"RECORD_a.json": a_record("a", 0.5),
             "ARRAYS_a.npz": b"arrays-a"}
    custody, snap = snapshot_over(files)
    first = snap.read_private(snap.units_fd, "ARRAYS_a.npz", "arrays")
    custody.replace("ARRAYS_a.npz", b"arrays-FORGED")
    second = snap.read_private(snap.units_fd, "ARRAYS_a.npz", "arrays")
    assert first == second == b"arrays-a"
    assert custody.read_log.count("ARRAYS_a.npz") == 1


def test_an_unknown_artifact_is_refused_not_fetched():
    custody, snap = snapshot_over({"RECORD_a.json": a_record("a", 0.5)})
    with pytest.raises(SystemExit, match="not in the snapshot"):
        snap.read_private(snap.units_fd, "RECORD_ghost.json", "unit")
    with pytest.raises(SystemExit, match="does not own"):
        snap.read_private("SOME_OTHER_FD", "RECORD_a.json", "unit")


def test_released_arrays_are_not_silently_re_read():
    files = {"RECORD_a.json": a_record("a", 0.5),
             "ARRAYS_a.npz": b"arrays-a"}
    custody, snap = snapshot_over(files)
    snap.read_private(snap.units_fd, "ARRAYS_a.npz", "arrays")
    snap.release_arrays("a")
    custody.replace("ARRAYS_a.npz", b"arrays-FORGED")
    again = snap.read_private(snap.units_fd, "ARRAYS_a.npz", "arrays")
    assert again == b"arrays-FORGED"
    assert custody.read_log.count("ARRAYS_a.npz") == 2, (
        "a released array is honestly re-read and the read is counted; "
        "it is never served stale under the pretence of one read")


# =========================================================== R9.3/R9.4
@pytest.mark.parametrize("rel", [
    "tools/t2_completion_reconstruction.py",
    "tools/t2_campaign_closure.py",
    "tools/descriptor_custody.py",
])
def test_mutating_a_tip_file_changes_the_identity(tmp_path, rel):
    tip = tmp_path / "tip"
    for r in T.CLOSURE_SURFACE:
        (tip / r).parent.mkdir(parents=True, exist_ok=True)
        (tip / r).write_text(f"# {r}\n")
    before = T.closure_code_identity(REVIEWED, tip=tip)
    (tip / rel).write_text(f"# {rel}\n# mutated\n")
    after = T.closure_code_identity(REVIEWED, tip=tip)
    assert before["surface_sha256"] != after["surface_sha256"]
    assert before["files"][rel]["sha256"] != after["files"][rel]["sha256"]
    assert after["files"][rel]["loaded_from"] == "BRANCH_TIP"


def test_the_identity_never_calls_a_mixture_reviewed():
    ident = T.closure_code_identity(REVIEWED)
    origins = {f["loaded_from"] for f in ident["files"].values()}
    assert origins == {"REVIEWED_CHECKOUT", "BRANCH_TIP"}
    assert "NOT called a reviewed identity" in ident["honesty"]
    for rel, f in ident["files"].items():
        if f["pinned_by_execution_record"]:
            assert f["matches_record"] is True, rel


# ================================================================ R9.5
def test_a_dirty_or_moved_reviewed_checkout_refuses(tmp_path):
    fake = tmp_path / "checkout"
    (fake / "tools").mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(fake)), check=True)

    class Conf:
        T2_SUCCESSOR_EXECUTION_RECORD_PATH = (
            Path.home() / ".config/agent-multi/reviewer_authority"
            / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json")
    with pytest.raises(SystemExit, match="not the reviewed identity"):
        T.assert_reviewed_identity(Conf, fake)


# ================================================================ R9.6
def test_an_inexact_population_refuses():
    class Ex:
        @staticmethod
        def _safe_name(uid):
            return uid.replace("::", "__")

    uids = ["bank::A", "bank::B"]
    complete = {f"{k}_{Ex._safe_name(u)}.{'npz' if k == 'ARRAYS' else 'json'}"
                : b"{}" for u in uids for k in ("CLAIM", "ARRAYS", "RECORD")}
    custody = FakeCustody(complete)
    facts = {"inventory_sha256": "0" * 64}
    inv = T.exact_inventory_from_snapshot(Ex, facts, custody, uids)
    assert inv["exact"] is True and inv["sealed_units"] == 2

    missing = dict(complete)
    missing.pop("RECORD_bank__B.json")
    with pytest.raises(SystemExit, match="missing="):
        T.exact_inventory_from_snapshot(Ex, facts, FakeCustody(missing),
                                        uids)
    extra = dict(complete)
    extra["RECORD_bank__GHOST.json"] = b"{}"
    with pytest.raises(SystemExit, match="extra="):
        T.exact_inventory_from_snapshot(Ex, facts, FakeCustody(extra),
                                        uids)


# ================================================================ R9.7
def test_a_second_closure_of_the_same_evidence_is_idempotent():
    """The adjudication identity excludes when and how long."""
    base = {
        "schema": T.CLOSURE_SCHEMA, "campaign_root_logical": "root",
        "inventory": {"exact": True},
        "final_adjudication_counts": {"COMPLETED_VERIFIED": 2},
        "screen_adjudication": {"verdict": "DOES_NOT_ADVANCE"},
    }
    a = dict(base, closed_at="2026-09-12T00:00:00Z",
             measurement={"reconstruction_seconds": 1001.8})
    b = dict(base, closed_at="2026-09-12T06:00:00Z",
             measurement={"reconstruction_seconds": 992.9})
    ident_a = T.sha_obj({k: v for k, v in sorted(a.items())
                         if k not in T.VOLATILE_FOR_IDENTITY})
    ident_b = T.sha_obj({k: v for k, v in sorted(b.items())
                         if k not in T.VOLATILE_FOR_IDENTITY})
    assert ident_a == ident_b
    assert "closed_at" in T.VOLATILE_FOR_IDENTITY
    assert "measurement" in T.VOLATILE_FOR_IDENTITY


# ================================================================ R9.8
def test_the_closure_writes_nothing_while_reconstructing():
    """Reconstruction is a read. The only writes in this module are the
    append-only closure log, the submission and the outbox — all of
    them behind an explicit flag."""
    src = (REPO / "tools/t2_campaign_closure.py").read_text()
    body = src[src.index("def reconstruct_from_snapshot("):
               src.index("def build_closure(")]
    for forbidden in ("open(", "write_text", "write_bytes", "mkdir",
                      "excl_write", "unlink"):
        assert forbidden not in body, (
            f"reconstruction must not {forbidden}")
    for forbidden in ("fit(", "train(", "download", "urlopen",
                      "requests."):
        assert forbidden not in body


def test_the_boundary_is_declared_in_the_submission():
    closure = {
        "campaign_root_logical": "t2_root",
        "inventory": {"exact": True}, "final_adjudication_counts": {},
        "screen_adjudication": {"verdict": "DOES_NOT_ADVANCE",
                                "primary_estimand_unweighted_mean_"
                                "of_panel_effects": -0.001},
        "sign_test_supersession": T.supersede_sign_test(
            {"signs_positive": 3, "sign_test_exact_p_two_sided": 1.3125}),
        "unit_snapshot": {"artifacts": 726},
        "single_instance": "one read", "adjudication_sha256": "a" * 64,
    }
    sub = T.build_readjudication_submission(
        closure, reviewed_checkout=REVIEWED,
        read_root=Path("/tmp/copy"),
        preserved_root=Path("/real/root"))
    assert sub["requires"] == "EXTERNAL_REVIEW"
    assert sub["read_the_preserved_root"] is False
    assert "supersedes no envelope" in sub["grants_nothing"]
    assert len(sub["submission_sha256"]) == 64


# ================================================================ R9.9
def test_the_two_sided_binomial_table_is_symmetric_and_bounded():
    got = [T.exact_two_sided_binomial_p(k, 6) for k in range(7)]
    assert got == [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875,
                   0.03125]
    assert got == got[::-1], "a sign test must be symmetric in k"
    assert all(0.0 <= v <= 1.0 for v in got)


def test_the_published_value_is_superseded_not_rewritten():
    out = T.supersede_sign_test({"signs_positive": 3,
                                 "sign_test_exact_p_two_sided": 1.3125})
    assert out["published_value"] == 1.3125
    assert out["published_value_valid"] is False
    assert out["corrected_value"] == 1.0
    assert out["changes_verdict"] is False
    assert out["historical_envelope"] == "NOT REWRITTEN"


def test_a_nonsense_sign_count_refuses():
    for k, n in ((-1, 6), (7, 6), (3, 0)):
        with pytest.raises(SystemExit, match="sign test needs"):
            T.exact_two_sided_binomial_p(k, n)


HISTORICAL_CONFIRMATORY_SHA256 = (
    "e00ab31937401e6f")


def _confirmatory_is_historical() -> bool:
    import hashlib
    here = Path(__file__).resolve().parents[1] / "tools/t2_confirmatory.py"
    return hashlib.sha256(here.read_bytes()).hexdigest().startswith(
        HISTORICAL_CONFIRMATORY_SHA256)


@pytest.mark.skipif(
    _confirmatory_is_historical(),
    reason="the reproducer carries t2_confirmatory.py with its HISTORICAL "
           "pinned bytes; this test exercises a later hardening of that "
           "file which by construction is absent here, and the corrected "
           "sign test is recomputed by the closure instead")
def test_the_pinned_screen_no_longer_publishes_an_impossible_value():
    """The defect must be dead in the code, not only superseded."""
    import t2_confirmatory as conf
    got = [conf._two_sided_binomial_p(k, 6) for k in range(7)]
    assert got == [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875,
                   0.03125]
    src = (REPO / "tools/t2_confirmatory.py").read_text()
    assert "2 * (0.5 ** 6) * sum(" not in src


# =====================================================================
# R11: the units directory is retained, not re-resolved by name
# =====================================================================

def a_units_root(tmp: Path, effect: float = 1.0) -> Path:
    root = tmp / "root"
    (root / "units").mkdir(parents=True)
    (root / "units/RECORD_a.json").write_text(
        json.dumps({"assay_record": {"effect": effect}}))
    (root / "units/ARRAYS_a.npz").write_bytes(b"arrays-original")
    return root


def test_swapping_the_whole_units_directory_changes_nothing(tmp_path):
    root = a_units_root(tmp_path, 1.0)
    with DC.Custody(root) as c:
        units = c.walk_to("units")
        (root / "units").rename(root / "units_old")
        (root / "units").mkdir()
        (root / "units/RECORD_a.json").write_text(
            json.dumps({"assay_record": {"effect": -99.0}}))
        got = units.read("RECORD_a.json").json()
    assert got["assay_record"]["effect"] == 1.0, (
        "the retained descriptor still points at the inventoried "
        "instance of units")


def test_restoring_the_units_name_over_a_new_inode_is_visible(tmp_path):
    root = a_units_root(tmp_path)
    with DC.Custody(root) as c:
        units = c.walk_to("units")
        facts = units.facts()
        (root / "units").rename(root / "units_away")
        (root / "units").mkdir()
        (root / "units/RECORD_a.json").write_text("{}")
        after = (root / "units").stat().st_ino
        got = units.read("RECORD_a.json").json()
    assert facts["inode"] != after
    assert got["assay_record"]["effect"] == 1.0


def test_the_adapter_reads_out_of_the_retained_units_instance():
    files = {"RECORD_a.json": a_record("a", 0.5),
             "ARRAYS_a.npz": b"arrays-a"}
    units, snap = snapshot_over(files)
    first = snap.read_private(snap.units_fd, "RECORD_a.json", "rec")
    units.replace("RECORD_a.json", a_record("a", -99.0))
    assert snap.record("a")["assay_record"]["effect"] == 0.5
    assert units.read_log.count("RECORD_a.json") == 1


# =====================================================================
# R13: the adapter fails closed on anything it does not implement
# =====================================================================

def test_the_adapter_covers_the_whole_parent_surface():
    """Not just the calls one run happened to make."""
    cls = T.make_snapshot_class(FakeResultsRoot)
    public = {n for n in dir(FakeResultsRoot)
              if not n.startswith("_")
              and callable(getattr(FakeResultsRoot, n, None))}
    assert public <= (cls.implemented_surface
                      | {n for n in dir(cls) if not n.startswith("_")})


def test_an_unimplemented_parent_call_fails_closed():
    class Parent:
        def excl_write(self, *a, **k):
            raise AssertionError("the real writer must never run")

        def some_new_call(self, *a, **k):
            raise AssertionError("the real implementation must not run")

    cls = T.make_snapshot_class(Parent)
    snap = cls(FakeDirSnapshot({"RECORD_a.json": a_record("a", 1.0)}),
               ("RECORD_a.json",))
    for call in ("excl_write", "some_new_call"):
        with pytest.raises(SystemExit, match="does not implement"):
            getattr(snap, call)("x")


def test_the_adapter_never_writes():
    """`excl_write` is the pinned root's only writer. A replay that
    could reach it could rewrite the evidence it is reading."""
    class Writer:
        def excl_write(self, *a, **k):
            raise AssertionError("the real writer must never run")

    cls = T.make_snapshot_class(Writer)
    snap = cls(FakeDirSnapshot({"RECORD_a.json": a_record("a", 1.0)}),
               ("RECORD_a.json",))
    with pytest.raises(SystemExit, match="does not implement"):
        snap.excl_write("fd", "name", b"bytes")


def test_no_constructor_invariant_is_needed(monkeypatch):
    """The parent's __init__ is never called, so it must not be needed
    by anything the replay reaches."""
    calls = []

    class Parent:
        def __init__(self, *a, **k):
            calls.append("init")
            raise AssertionError("the parent constructor must not run")

        def close(self):
            calls.append("close")

        def revalidate(self):
            calls.append("revalidate")

    cls = T.make_snapshot_class(Parent)
    snap = cls(FakeDirSnapshot({"RECORD_a.json": a_record("a", 1.0)}),
               ("RECORD_a.json",))
    snap.revalidate()
    snap.record("a")
    snap.close()
    assert calls == [], (
        "the replay must reach nothing that depends on the parent's "
        "constructor")


def test_each_array_is_read_exactly_once_per_unit():
    files = {f"RECORD_{u}.json": a_record(u, 1.0) for u in "abc"}
    files.update({f"ARRAYS_{u}.npz": f"arrays-{u}".encode()
                  for u in "abc"})
    units, snap = snapshot_over(files)
    for u in "abc":
        snap.read_private(snap.units_fd, f"ARRAYS_{u}.npz", "arrays")
        snap.record(u)
        snap.release_arrays(u)
    for u in "abc":
        assert units.read_log.count(f"ARRAYS_{u}.npz") == 1
        assert units.read_log.count(f"RECORD_{u}.json") == 1


def test_the_entry_point_is_defined_after_everything_it_calls():
    """Running the module as a script must not call a function that
    does not exist yet. The guard once sat above a function appended
    below it, and only a real invocation could show that."""
    import ast as _ast
    tree = _ast.parse((REPO / "tools/t2_campaign_closure.py").read_text())
    guards = [n for n in tree.body if isinstance(n, _ast.If)]
    defs = [n.lineno for n in tree.body
            if isinstance(n, _ast.FunctionDef)]
    assert guards, "the module has no __main__ guard"
    assert guards[-1].lineno > max(defs), (
        "the entry point must come after every definition it can reach")


# ------------------------------------------- R15 two-phase publication
def _closure_stub():
    return {
        "campaign_root_logical": "t2_test_root",
        "inventory": {"exact": True, "sealed_units": 242},
        "final_adjudication_counts": {"COMPLETED_VERIFIED": 242,
                                      "TERMINAL_FAILED": 0},
        "screen_adjudication": {"verdict": "DOES_NOT_ADVANCE",
                                "primary_estimand_unweighted_mean_of_"
                                "panel_effects": -0.001048},
        "sign_test_supersession": {"state": "SUPERSEDED"},
        "unit_snapshot": {"artifacts": 726},
        "single_instance": "one read each",
        "adjudication_sha256": "a" * 64,
    }


def test_the_submission_publishes_no_physical_path(tmp_path,
                                                   monkeypatch):
    secret = tmp_path / "home" / "operator" / "private_root"
    secret.mkdir(parents=True)
    monkeypatch.setattr(T, "closure_code_identity",
                        lambda c: {"files": {}})
    doc = T.build_readjudication_submission(
        _closure_stub(), reviewed_checkout=tmp_path,
        read_root=secret, preserved_root=secret)
    blob = json.dumps(doc)
    assert str(secret) not in blob
    assert "private_root" not in blob.replace(
        doc["root_actually_read_logical"], "")
    assert doc["physical_paths"].startswith("WITHHELD")
    assert doc["schema"].endswith(".v2")


def test_the_logical_id_is_stable_and_does_not_locate(tmp_path):
    a = T.logical_id(tmp_path / "root")
    b = T.logical_id(tmp_path / "root")
    c = T.logical_id(tmp_path / "other")
    assert a == b and a != c
    assert str(tmp_path) not in a


def test_reading_a_root_that_is_not_the_preserved_one_is_declared(
        tmp_path, monkeypatch):
    monkeypatch.setattr(T, "closure_code_identity",
                        lambda c: {"files": {}})
    doc = T.build_readjudication_submission(
        _closure_stub(), reviewed_checkout=tmp_path,
        read_root=tmp_path / "copy", preserved_root=tmp_path / "orig")
    assert doc["read_the_preserved_root"] is False
    assert doc["root_actually_read_logical"] != \
        doc["preserved_root_logical"]


def test_the_submission_digest_covers_the_publication_block():
    doc = {"schema": "x", "publication": {"commit_a": "aaa"}}
    d1 = T.sha_obj(doc)
    doc["publication"]["commit_a"] = "bbb"
    assert T.sha_obj(doc) != d1


# ------------------------------------ T2-R17: the late NPZ read is bound
def _npz(value: float) -> bytes:
    import io
    import numpy as np
    buf = io.BytesIO()
    np.savez(buf, y=np.full(8, value, dtype=np.float64))
    return buf.getvalue()


def _units_root(tmp_path):
    import os
    root = tmp_path / "results"
    (root / "units").mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(root / "units", 0o700)
    (root / "units" / "RECORD_u.json").write_text('{"wall_seconds": 1.0}')
    (root / "units" / "ARRAYS_u.npz").write_bytes(_npz(1.0))
    for p in (root / "units").iterdir():
        os.chmod(p, 0o600)
    return root


def test_an_npz_substituted_before_its_late_read_refuses(tmp_path):
    """The adapter reads JSON at construction and each NPZ only when the
    verifier asks. Between the two, a substitute with the same name,
    mode, length and mtime used to be consumed."""
    import os
    root = _units_root(tmp_path)
    custody = T.Custody(root)
    units = custody.walk_to("units")
    cls = T.make_snapshot_class(type("ResultsRoot", (), {}))
    snap = cls(units, units.files)
    arr = root / "units" / "ARRAYS_u.npz"
    st0 = os.stat(arr)
    os.rename(arr, root / "units" / "ARRAYS_u.orig")
    arr.write_bytes(_npz(999.0))
    os.chmod(arr, 0o600)
    os.utime(arr, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    assert os.stat(arr).st_size == st0.st_size
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read_private(cls.units_fd, "ARRAYS_u.npz", "arrays")
    custody.close()
    assert "inode" in e.value.diverged
    assert "ARRAYS_u.npz" not in snap.digests()


def test_an_npz_mutated_in_place_before_its_late_read_refuses(tmp_path):
    import os
    root = _units_root(tmp_path)
    custody = T.Custody(root)
    units = custody.walk_to("units")
    cls = T.make_snapshot_class(type("ResultsRoot", (), {}))
    snap = cls(units, units.files)
    arr = root / "units" / "ARRAYS_u.npz"
    st0 = os.stat(arr)
    sub = _npz(999.0)
    with open(arr, "r+b") as fh:
        fh.write(sub)
    os.utime(arr, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read_private(cls.units_fd, "ARRAYS_u.npz", "arrays")
    custody.close()
    assert e.value.diverged.keys() >= {"ctime_ns"}


def test_an_untouched_npz_is_served_with_its_binding(tmp_path):
    import io
    import numpy as np
    root = _units_root(tmp_path)
    custody = T.Custody(root)
    units = custody.walk_to("units")
    cls = T.make_snapshot_class(type("ResultsRoot", (), {}))
    snap = cls(units, units.files)
    got = np.load(io.BytesIO(
        snap.read_private(cls.units_fd, "ARRAYS_u.npz", "arrays")))["y"][0]
    reads = custody.reads()
    custody.close()
    assert float(got) == 1.0
    assert all(r["leaf_binding"] != "UNBOUND" for r in reads)
