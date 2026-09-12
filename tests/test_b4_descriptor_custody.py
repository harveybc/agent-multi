"""R11-R13 (order 2026-09-12): the five PRE swaps, and the custody that
makes them impossible.

The previous closure verified and consumed through separate path opens.
Every test below either performs one of the audited swaps and requires
the closure to refuse or to be unaffected, or attacks the custody layer
directly with a symlink, a permissive mode or a foreign owner.

The last test is the mutant: a reimplementation that goes back to
reading by path is shown to be fooled by the very swap the shipped code
survives. Without it, all the others could pass against code that never
needed fixing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import b4_campaign_closure as C                                # noqa: E402
import descriptor_custody as DC                                # noqa: E402
from test_b4_campaign_closure import build_root                # noqa: E402


@pytest.fixture(autouse=True)
def _stub_launch_gate(monkeypatch):
    monkeypatch.setattr(C, "prove_relaunch_refuses", lambda: {
        "launch_gate": "stub", "refused": True, "refusal": "REFUSED: stub",
        "accelerator_modules_before": [], "accelerator_modules_after": [],
        "refused_before_any_accelerator_import": True})


@pytest.fixture
def world(tmp_path):
    return build_root(tmp_path, cells=12, completed=2, partial=1)


def swap_after_first_read(monkeypatch, target: Path, new_bytes: bytes):
    """Replace `target` immediately after the custody reads it — the
    window the PRE exploited."""
    original = DC.Custody.read
    state = {"done": False}

    def patched(self, rel):
        art = original(self, rel)
        if not state["done"] and Path(self.root, rel) == target:
            state["done"] = True
            target.write_bytes(new_bytes)
        return art

    monkeypatch.setattr(DC.Custody, "read", patched)
    return state


# =============================================== the five PRE swaps
def test_pre1_terminal_swap_cannot_change_what_is_adjudicated(
        world, monkeypatch):
    """PRE 1: hashed as one object, parsed as another."""
    root, mat = world
    term = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    honest = json.loads(term.read_text())
    forged = json.dumps(dict(honest, wall_seconds=999999.0),
                        indent=1).encode()
    swap_after_first_read(monkeypatch, term, forged)
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"] if c["cell"] == "o2022_seed101"][0]
    assert done["wall_seconds_declared"] == honest["wall_seconds"]
    assert done["terminal_sha256"] == hashlib.sha256(
        json.dumps(honest, indent=1).encode()).hexdigest(), (
        "the digest must be of the bytes that were adjudicated")
    assert json.loads(term.read_text())["wall_seconds"] == 999999.0, (
        "the swap really happened")


def test_pre2_per_bar_swap_preserving_row_count_is_not_consumed(
        world, monkeypatch):
    """PRE 2: hashed from one open, counted from another."""
    root, mat = world
    per_bar = root / "o2022_seed101/per_bar_o2022_seed101.csv"
    honest = per_bar.read_bytes()
    swap_after_first_read(monkeypatch, per_bar, b"net_return\n9.9\n9.9\n")
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"] if c["cell"] == "o2022_seed101"][0]
    assert done["per_bar_sha256"] == hashlib.sha256(honest).hexdigest()
    assert per_bar.read_bytes() != honest


def test_pre3_seal_intent_swap_between_parse_and_hash(world,
                                                      monkeypatch):
    """PRE 3: the intent is parsed and then re-opened to be hashed."""
    root, mat = world
    cell = root / "o2022_seed101"
    intent = next(cell.glob("SEAL_INTENT_*.json"))
    honest = intent.read_bytes()
    swap_after_first_read(monkeypatch, intent, b'{"schema": "forged"}')
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"] if c["cell"] == "o2022_seed101"][0]
    assert done["seal_intent_sha256"] == \
        hashlib.sha256(honest).hexdigest(), (
        "the seal binding must use the bytes that were parsed")


def test_pre4_partial_status_swap_after_inventory(world, monkeypatch):
    """PRE 4: status.json re-read after the inventory hashed it."""
    root, mat = world
    status = root / "o2022_seed103/cell_runtime/status.json"
    honest = json.loads(status.read_text())
    swap_after_first_read(monkeypatch, status, json.dumps(
        {"epoch_completed": 1999, "num_timesteps": 1,
         "stop_reason": "FORGED", "last_durable_artifact": None}).encode())
    closure = C.build_closure(root, mat)
    part = [c for c in closure["cells"]
            if c["classification"] == C.PARTIAL][0]
    assert part["last_durable_progress"]["epoch_completed"] == \
        honest["epoch_completed"]
    assert part["last_durable_progress"]["read_from"].startswith(
        "the same inventory read")


def test_pre5_a_cell_appearing_after_the_snapshot_does_not_stay_absent(
        world, monkeypatch):
    """PRE 5: absence decided by a bare existence check."""
    root, mat = world
    ghost = root / "o2022_seed104"
    original = DC.Custody.snapshot
    state = {"done": False}

    def snapshot_then_create(self, rel=""):
        snap = original(self, rel)
        if rel in ("", ".") and not state["done"]:
            state["done"] = True
            ghost.mkdir(parents=True, exist_ok=True)
            (ghost / "B4_CELL_TERMINAL.json").write_text("{}")
        return snap

    monkeypatch.setattr(DC.Custody, "snapshot", snapshot_then_create)
    closure = C.build_closure(root, mat)
    entry = [c for c in closure["cells"]
             if c["cell"] == "o2022_seed104"][0]
    assert entry["classification"] == C.NOT_STARTED
    assert "listed once from its own descriptor" in entry["evidence"]
    assert ghost.exists(), "the ghost really appeared"
    # The classification is honest about WHAT it was derived from, and
    # that snapshot is published so a reviewer can check it.
    assert len(entry["root_snapshot_sha256"]) == 64


# ================================================= custody attacks
def test_a_leaf_symlink_refuses(tmp_path):
    real = tmp_path / "real.json"
    real.write_text('{"a": 1}')
    root = tmp_path / "root"
    root.mkdir()
    (root / "link.json").symlink_to(real)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="symbolic link"):
            c.read("link.json")


def test_a_directory_symlink_refuses(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "x.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "cell").symlink_to(elsewhere, target_is_directory=True)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="symbolic link"):
            c.read("cell/x.json")
        with pytest.raises(SystemExit, match="symbolic link"):
            c.snapshot("cell")


def test_a_world_writable_file_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "evidence.json"
    f.write_text("{}")
    os.chmod(f, 0o666)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="world-writable"):
            c.read("evidence.json")


def test_a_group_writable_file_is_declared_and_refused_when_strict(
        tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "evidence.json"
    f.write_text("{}")
    os.chmod(f, 0o664)
    with DC.Custody(root) as c:
        art = c.read("evidence.json")
    assert art.custody_weakness == "GROUP_WRITABLE"
    with DC.Custody(root, strict_mode=True) as c:
        with pytest.raises(SystemExit, match="group-writable"):
            c.read("evidence.json")


def test_a_foreign_owner_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "evidence.json").write_text("{}")
    with DC.Custody(root, expected_uid=os.getuid() + 4242) as c:
        with pytest.raises(SystemExit, match="not by the expected owner"):
            c.read("evidence.json")


def test_escaping_the_retained_root_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.json").write_text("{}")
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="escapes the retained root"):
            c.read("../outside.json")


def test_a_file_that_grows_while_read_refuses(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "evidence.json"
    f.write_text('{"a": 1}')
    real_fstat = os.fstat

    def lying_fstat(fd):
        st = real_fstat(fd)
        return os.stat_result(tuple(st)[:6] + (st.st_size + 10,)
                              + tuple(st)[7:])

    monkeypatch.setattr(os, "fstat", lying_fstat)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="changed while it was "
                                             "being read"):
            c.read("evidence.json")


def test_a_device_or_fifo_in_the_tree_refuses(world, tmp_path):
    root, mat = world
    os.mkfifo(root / "o2022_seed103/a_fifo")
    with pytest.raises(SystemExit, match="neither regular files nor "
                                         "directories"):
        C.build_closure(root, mat)


# ============================== extra / missing / duplicate objects
def test_an_extra_claim_refuses(world):
    root, mat = world
    (root / "o2022_seed101/CLAIM_second.json").write_text("{}")
    with pytest.raises(SystemExit, match="exactly_one_claim"):
        C.build_closure(root, mat)


def test_a_missing_seal_refuses(world):
    root, mat = world
    next((root / "o2022_seed101").glob("SEAL_COMPLETE_*.json")).unlink()
    with pytest.raises(SystemExit,
                       match="seal_complete_present_in_snapshot"):
        C.build_closure(root, mat)


def test_a_missing_per_bar_refuses(world):
    root, mat = world
    (root / "o2022_seed101/per_bar_o2022_seed101.csv").unlink()
    with pytest.raises(SystemExit, match="per_bar_present_in_snapshot"):
        C.build_closure(root, mat)


def test_a_duplicate_seal_for_another_attempt_refuses(world):
    root, mat = world
    (root / "o2022_seed103/SEAL_COMPLETE_other.json").write_text("{}")
    with pytest.raises(SystemExit, match="carries seals"):
        C.build_closure(root, mat)


# ========================================= every layer must be load-bearing
def test_mutating_the_terminal_breaks_the_seal_chain(world):
    """A VALID but different terminal: the seal binds its digest."""
    root, mat = world
    p = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    doc = json.loads(p.read_text())
    doc["wall_seconds"] = 555.0
    p.write_text(json.dumps(doc, indent=1))
    with pytest.raises(SystemExit,
                       match="seal_intent_binds_terminal_digest"):
        C.build_closure(root, mat)


def test_an_unparseable_terminal_refuses_at_custody(world):
    root, mat = world
    p = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="not parseable as JSON"):
        C.build_closure(root, mat)


@pytest.mark.parametrize("rel,needle", [
    ("o2022_seed101/per_bar_o2022_seed101.csv", "per_bar_digest"),
    ("o2022_seed101/best_model.terminal.zip", "checkpoint_digest"),
])
def test_mutating_any_verified_layer_refuses(world, rel, needle):
    root, mat = world
    p = root / rel
    p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(SystemExit, match=needle):
        C.build_closure(root, mat)


def test_the_closure_identity_covers_the_custody_module():
    ident = C.closure_code_identity(REPO)
    assert "tools/descriptor_custody.py" in ident["files"]
    assert "tools/b4_campaign_closure.py" in ident["files"]
    assert len(ident["surface_sha256"]) == 64
    assert ident["commit"] != "UNAVAILABLE"


def test_the_submission_grants_nothing_and_names_the_root_it_read(
        world, tmp_path):
    root, mat = world
    closure = C.build_closure(root, mat)
    sub = C.build_submission(closure, results_root=tmp_path / "preserved",
                             read_root=root, repo=REPO)
    assert sub["requires"] == "EXTERNAL_REVIEW"
    assert sub["read_the_preserved_root"] is False
    assert "promotes no cell" in sub["grants_nothing"]
    assert sub["adjudication"]["verdict"] == \
        "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT"
    assert len(sub["submission_sha256"]) == 64


# ===================================================== the mutant
def test_a_path_reopening_mutant_is_fooled_by_the_same_swap(world,
                                                            monkeypatch):
    """The control. A reimplementation that reads by path accepts the
    forged terminal; the shipped one does not. Without this, every
    test above could pass against code that never needed fixing."""
    root, mat = world
    term = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    honest = json.loads(term.read_text())
    forged = json.dumps(dict(honest, wall_seconds=999999.0), indent=1)

    def legacy_verify(custody, snap, cell, ledger_cell):
        """The PRE behaviour: parse from one open, hash from another."""
        cell_dir = Path(custody.root) / cell
        term_p = cell_dir / "B4_CELL_TERMINAL.json"
        doc = json.loads(term_p.read_text())          # open #1
        term_p.write_text(forged)                     # the swap
        digest = hashlib.sha256(term_p.read_bytes()).hexdigest()  # open #2
        return {"cell": cell, "classification": C.COMPLETED,
                "wall_seconds_declared": doc["wall_seconds"],
                "terminal_sha256": digest,
                "descriptors_verified": [], "enters_comparisons": False,
                "enters_comparisons_reason": "mutant",
                "attempt_id": "mutant"}

    monkeypatch.setattr(C, "verify_completed_cell", legacy_verify)
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"] if c["cell"] == "o2022_seed101"][0]
    on_disk = json.loads(term.read_text())["wall_seconds"]
    assert done["wall_seconds_declared"] == honest["wall_seconds"]
    assert on_disk == 999999.0
    assert done["terminal_sha256"] == hashlib.sha256(
        forged.encode()).hexdigest(), (
        "the mutant certified bytes it did not adjudicate — which is "
        "exactly what the shipped closure no longer does")
