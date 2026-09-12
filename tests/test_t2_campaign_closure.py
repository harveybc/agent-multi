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
class FakeCustody:
    """A custody that serves prepared bytes and COUNTS its reads."""

    def __init__(self, files: dict[str, bytes]):
        self.root = Path("/fake")
        self._files = dict(files)
        self.read_log: list[str] = []

    def read(self, rel: str):
        self.read_log.append(rel)
        name = rel.split("/")[-1]
        if name not in self._files:
            raise DC.CustodyRefusal(f"{rel}: does not exist")
        payload = self._files[name]

        class _St:
            st_uid = 0
            st_mode = 0o100644
            st_ino = 1
            st_dev = 1
            st_mtime_ns = 0
        return DC.Artifact(rel, payload, _St())

    def snapshot(self, rel: str = ""):
        return DC.DirSnapshot(rel, tuple(sorted(self._files)), (), ())

    def replace(self, name: str, payload: bytes):
        self._files[name] = payload

    def close(self):
        pass


class FakeResultsRoot:
    """Stands in for the pinned ResultsRoot in the subclass factory."""


def snapshot_over(files: dict[str, bytes]):
    cls = T.make_snapshot_class(FakeResultsRoot)
    custody = FakeCustody(files)
    snap = cls(custody, tuple(sorted(files)), path=Path("/fake"))
    return custody, snap


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
    assert custody.read_log.count("units/RECORD_a.json") == 1, (
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
    assert custody.read_log.count("units/ARRAYS_a.npz") == 1


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
    assert custody.read_log.count("units/ARRAYS_a.npz") == 2, (
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


def test_the_pinned_screen_no_longer_publishes_an_impossible_value():
    """The defect must be dead in the code, not only superseded."""
    import t2_confirmatory as conf
    got = [conf._two_sided_binomial_p(k, 6) for k in range(7)]
    assert got == [0.03125, 0.21875, 0.6875, 1.0, 0.6875, 0.21875,
                   0.03125]
    src = (REPO / "tools/t2_confirmatory.py").read_text()
    assert "2 * (0.5 ** 6) * sum(" not in src
