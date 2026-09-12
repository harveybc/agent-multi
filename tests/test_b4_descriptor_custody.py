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
    """Replace `target` immediately after the snapshot reads it — the
    window the PRE exploited."""
    original = DC.DirSnapshot.read
    state = {"done": False}

    def patched(self, name):
        art = original(self, name)
        if not state["done"] and target.name == name:
            state["done"] = True
            target.write_bytes(new_bytes)
        return art

    monkeypatch.setattr(DC.DirSnapshot, "read", patched)
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
    original = DC.Custody.root_snapshot
    state = {"done": False}

    def snapshot_then_create(self):
        snap = original(self)
        if not state["done"]:
            state["done"] = True
            ghost.mkdir(parents=True, exist_ok=True)
            (ghost / "B4_CELL_TERMINAL.json").write_text("{}")
        return snap

    monkeypatch.setattr(DC.Custody, "root_snapshot",
                        snapshot_then_create)
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
        with pytest.raises(SystemExit, match="not in the inventory"):
            c.root_snapshot().read("link.json")


def test_a_directory_symlink_refuses(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "x.json").write_text("{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "cell").symlink_to(elsewhere, target_is_directory=True)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="not a directory in the "
                                             "inventory"):
            c.walk_to("cell")


def test_a_world_writable_file_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "evidence.json"
    f.write_text("{}")
    os.chmod(f, 0o666)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="world-writable"):
            c.root_snapshot().read("evidence.json")


def test_a_group_writable_file_is_declared_and_refused_when_strict(
        tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    f = root / "evidence.json"
    f.write_text("{}")
    os.chmod(f, 0o664)
    with DC.Custody(root) as c:
        art = c.root_snapshot().read("evidence.json")
    assert art.custody_weakness == "GROUP_WRITABLE"
    with DC.Custody(root, strict_mode=True) as c:
        with pytest.raises(SystemExit, match="group-writable"):
            c.root_snapshot().read("evidence.json")


def test_a_foreign_owner_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "evidence.json").write_text("{}")
    with DC.Custody(root, expected_uid=os.getuid() + 4242) as c:
        with pytest.raises(SystemExit, match="not by the expected owner"):
            c.root_snapshot().read("evidence.json")


def test_escaping_the_retained_root_refuses(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.json").write_text("{}")
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="escapes the retained root"):
            c.walk_to("../outside")


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

    with DC.Custody(root) as c:
        snap = c.root_snapshot()
        monkeypatch.setattr(os, "fstat", lying_fstat)
        # R20: the lying fstat is now caught one step earlier, at the
        # comparison against the inventory, which is stricter than the
        # size-versus-bytes check this test was written for. Either
        # refusal returns no value.
        with pytest.raises(SystemExit, match="changed while it was "
                                             "being read|leaf identity "
                                             "diverged"):
            snap.read("evidence.json")


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
    # R17.4: no physical path anywhere in the public artifact.
    blob = json.dumps(sub)
    assert "/home/" not in blob and "/Users/" not in blob, (
        "a versioned submission must carry logical ids, not this "
        "host's topology")
    assert sub["root_actually_read_logical"]
    assert sub["preserved_root_logical"]
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

    def legacy_verify(snap, cell, ledger_cell):
        """The PRE behaviour: parse from one open, hash from another."""
        cell_dir = root / cell
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


# =====================================================================
# R15-R18: a photograph must retain the instance photographed
# =====================================================================
#
# The audit's counterexample: `DirSnapshot` kept only NAMES and closed
# the directory descriptor, so every later read re-walked the path. A
# photographed cell directory could be renamed, a different directory
# put in its place, and the read consumed the replacement — while the
# 45 tests above stayed green, because not one of them replaced an
# existing DIRECTORY.

def a_cell(tmp: Path, value: float = 1.0) -> Path:
    root = tmp / "root"
    (root / "cell").mkdir(parents=True)
    (root / "cell/terminal.json").write_text(
        json.dumps({"wall_seconds": value}))
    return root


def test_replacing_a_photographed_directory_consumes_the_retained_one(
        tmp_path):
    root = a_cell(tmp_path, 1.0)
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        (root / "cell").rename(root / "cell_moved")
        (root / "cell").mkdir()
        (root / "cell/terminal.json").write_text(
            json.dumps({"wall_seconds": 999.0}))
        got = cell.read("terminal.json").json()
    assert got["wall_seconds"] == 1.0, (
        "the retained descriptor still points at the instance that was "
        "inventoried")
    assert json.loads(
        (root / "cell/terminal.json").read_text())["wall_seconds"] \
        == 999.0, "the replacement really happened"


def test_replacing_an_intermediate_directory_changes_nothing(tmp_path):
    root = tmp_path / "root"
    (root / "cell/runtime").mkdir(parents=True)
    (root / "cell/runtime/status.json").write_text(
        json.dumps({"epoch_completed": 1}))
    with DC.Custody(root) as c:
        runtime = c.walk_to("cell/runtime")
        (root / "cell").rename(root / "cell_old")
        (root / "cell/runtime").mkdir(parents=True)
        (root / "cell/runtime/status.json").write_text(
            json.dumps({"epoch_completed": 1999}))
        got = runtime.read("status.json").json()
    assert got["epoch_completed"] == 1, (
        "every intermediate component is retained, so the whole "
        "subtree cannot be exchanged under the snapshot")


def test_restore_after_swap_is_visible_in_the_published_facts(tmp_path):
    root = a_cell(tmp_path, 1.0)
    original_ino = (root / "cell").stat().st_ino
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        facts = cell.facts()
        (root / "cell").rename(root / "cell_away")
        (root / "cell").mkdir()
        (root / "cell/terminal.json").write_text(
            json.dumps({"wall_seconds": 2.0}))
        got = cell.read("terminal.json").json()
    assert facts["inode"] == original_ino
    assert facts["device"] and facts["inode"], (
        "the snapshot publishes device and inode, so name equality is "
        "no longer the only thing binding a read to a listing")
    assert (root / "cell").stat().st_ino != facts["inode"]
    assert got["wall_seconds"] == 1.0


def test_the_artifact_records_the_directory_it_came_from(tmp_path):
    root = a_cell(tmp_path)
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        art = cell.read("terminal.json")
        assert art.dir_inode == cell.inode
        assert art.dir_device == cell.device
        assert art.facts()["dir_inode"] == cell.inode


def test_a_path_with_a_slash_cannot_be_read_from_a_snapshot(tmp_path):
    root = a_cell(tmp_path)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="not a name in this "
                                             "directory"):
            c.root_snapshot().read("cell/terminal.json")


def test_a_file_outside_the_inventory_refuses(tmp_path):
    root = a_cell(tmp_path)
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        (root / "cell/appeared.json").write_text("{}")
        with pytest.raises(SystemExit, match="not in the inventory"):
            cell.read("appeared.json")


def test_reading_after_close_refuses(tmp_path):
    root = a_cell(tmp_path)
    c = DC.Custody(root)
    cell = c.walk_to("cell")
    c.close()
    with pytest.raises(SystemExit, match="was closed"):
        cell.read("terminal.json")


def test_descriptors_close_on_success_and_on_exception(tmp_path):
    root = a_cell(tmp_path)
    c = DC.Custody(root)
    c.walk_to("cell")
    assert c.open_descriptors() == 2
    c.close()
    assert c.open_descriptors() == 0

    try:
        with DC.Custody(root) as c2:
            c2.walk_to("cell")
            assert c2.open_descriptors() == 2
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert c2.open_descriptors() == 0, (
        "an exception must not leak a directory descriptor")


def test_a_world_writable_directory_refuses(tmp_path):
    root = tmp_path / "root"
    (root / "cell").mkdir(parents=True)
    (root / "cell/x.json").write_text("{}")
    os.chmod(root / "cell", 0o777)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit, match="directory mode .* "
                                             "world-writable"):
            c.walk_to("cell")


def test_a_group_writable_directory_is_declared_and_strict_refuses(
        tmp_path):
    root = tmp_path / "root"
    (root / "cell").mkdir(parents=True)
    (root / "cell/x.json").write_text("{}")
    os.chmod(root / "cell", 0o775)
    with DC.Custody(root) as c:
        assert c.walk_to("cell").custody_weakness == "GROUP_WRITABLE"
    with DC.Custody(root, strict_mode=True) as c:
        with pytest.raises(SystemExit, match="group-writable"):
            c.walk_to("cell")


def test_a_foreign_owned_directory_refuses(tmp_path):
    root = tmp_path / "root"
    (root / "cell").mkdir(parents=True)
    with DC.Custody(root, expected_uid=os.getuid() + 4242) as c:
        with pytest.raises(SystemExit,
                           match="directory owned by uid"):
            c.walk_to("cell")


def test_the_closure_publishes_the_directory_instance(world):
    root, mat = world
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"]
            if c["classification"] == C.COMPLETED][0]
    inst = done["custody"]["directory_instance"]
    assert inst["device"] and inst["inode"]
    part = [c for c in closure["cells"]
            if c["classification"] == C.PARTIAL][0]
    assert part["directory_instances"], (
        "every directory the inventory descended into is published")
    for d in part["directory_instances"]:
        assert d["inode"] and d["device"]
    assert closure["custody"]["retained_descriptors_during_the_run"] > 1


# ------------------------------------------------ the guard mutants
def test_mutant_without_retention_readmits_the_directory_attack(
        tmp_path, monkeypatch):
    """Remove the retention and the exact PRE attack comes back."""
    root = a_cell(tmp_path, 1.0)

    def unretained_read(self, name):
        # the pre-R16 behaviour: resolve the path again from the root
        return json.loads((root / self.rel / name).read_text())

    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        (root / "cell").rename(root / "cell_moved")
        (root / "cell").mkdir()
        (root / "cell/terminal.json").write_text(
            json.dumps({"wall_seconds": 999.0}))
        monkeypatch.setattr(DC.DirSnapshot, "read", unretained_read)
        got = cell.read("terminal.json")
    assert got["wall_seconds"] == 999.0, (
        "without retention the replacement is consumed — which is "
        "exactly what the shipped code no longer does")


def test_mutant_without_directory_identity_hides_the_swap(tmp_path,
                                                          monkeypatch):
    root = a_cell(tmp_path)
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        honest = cell.facts()
        monkeypatch.setattr(
            DC.DirSnapshot, "facts",
            lambda self: {"rel": self.rel, "files": sorted(self.files)})
        blind = cell.facts()
    assert {"device", "inode"} <= set(honest)
    assert not {"device", "inode"} & set(blind), (
        "drop device and inode and a replaced directory becomes "
        "indistinguishable from the one that was photographed")


def test_mutant_without_owner_and_mode_checks_admits_a_weak_directory(
        tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "cell").mkdir(parents=True)
    os.chmod(root / "cell", 0o777)
    with DC.Custody(root) as c:
        with pytest.raises(SystemExit):
            c.walk_to("cell")
    monkeypatch.setattr(DC, "WORLD_WRITE", 0)
    with DC.Custody(root) as c:
        snap = c.walk_to("cell")
    assert snap.rel == "cell", (
        "with the mode check removed a world-writable directory is "
        "accepted as evidence")


# ------------------------------------------------- the strict bundle
def test_a_strict_read_bundle_is_built_without_touching_the_original(
        tmp_path):
    root = tmp_path / "root"
    (root / "cell").mkdir(parents=True)
    f = root / "cell/terminal.json"
    f.write_text(json.dumps({"wall_seconds": 1.0}))
    os.chmod(f, 0o664)
    os.chmod(root / "cell", 0o775)
    before = (f.stat().st_mode, f.read_bytes())

    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        doc = c.export_bundle(tmp_path / "bundle", snapshots=[cell],
                              provenance="group-writable legacy root")

    assert before == (f.stat().st_mode, f.read_bytes()), (
        "the original's bytes and mode are untouched")
    copied = tmp_path / "bundle/cell/terminal.json"
    assert copied.read_bytes() == before[1]
    assert oct(copied.stat().st_mode)[-3:] == "400"
    assert doc["originals"].startswith("UNTOUCHED")
    assert "group-writable" in doc["provenance"]
    with DC.Custody(tmp_path / "bundle", strict_mode=True) as c:
        c.walk_to("cell").read("terminal.json")


def test_the_v3_submission_publishes_that_every_read_leaf_was_bound(world,
                                                                    tmp_path):
    """B4-R21: the submission says how many reads were leaf-bound, and
    it must be all of them."""
    results_root, mat_root = world
    closure = C.build_closure(results_root, mat_root)
    lb = closure["custody"]["leaf_binding"]
    assert lb["contract"] == DC.CONTRACT
    assert lb["reads_total"] > 0 and lb["reads_bound"] == lb["reads_total"]
    sub = C.build_submission(closure, results_root=results_root,
                             read_root=results_root)
    assert sub["schema"] == "agent_multi.b4_readjudication_submission.v3"
    assert sub["custody"]["leaf_binding"] == lb
