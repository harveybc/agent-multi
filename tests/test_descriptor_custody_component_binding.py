"""B4-R26 / C88 / T2-R23: directories are bound like leaves, listings
are bracketed, and JSON is strict.

Byte-identical in predictor, B4 and T2; runs against each repository's
own tools/descriptor_custody.py. Every attack keeps the directory's NAME
and changes the instance or its contents; every mutant removes one
guard and shows its attack getting through.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import descriptor_custody as DC  # noqa: E402


def tree(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    (root / "cell").mkdir(parents=True)
    (root / "cell" / "terminal.json").write_text('{"wall": 1}')
    return root


def replace_cell(root: Path):
    os.rename(root / "cell", root / "cell.original")
    (root / "cell").mkdir()
    (root / "cell" / "terminal.json").write_text('{"wall": 9}')


def fds() -> int:
    return len(os.listdir("/proc/self/fd"))


# ------------------------------------------------------------ positive
def test_an_untouched_tree_publishes_directory_bindings(tmp_path):
    root = tree(tmp_path)
    with DC.Custody(root) as c:
        cell = c.walk_to("cell")
        assert cell.read("terminal.json").json() == {"wall": 1}
        b = cell.facts()["directory_binding"]
        assert b["contract"] == DC.CONTRACT
        assert b["fstat_enumeration_before"] == b["fstat_enumeration_after"]
        assert b["inventoried_by_parent"] == b["fstat_open"]
        root_b = c.root_snapshot().facts()["directory_binding"]
        assert root_b["inventoried_by_parent"] == "ROOT"
        assert len(b["entries_sha256"]) == 64


def test_the_inventory_keeps_facts_for_directories_and_other_objects(tmp_path):
    root = tree(tmp_path)
    os.symlink(root / "cell", root / "link")
    with DC.Custody(root) as c:
        snap = c.root_snapshot()
        assert set(snap.leaf_inventory("cell")) == set(DC.LEAF_IDENTITY_FIELDS)
        assert "link" in snap.others
        assert set(snap.leaf_inventory("link")) == set(DC.LEAF_IDENTITY_FIELDS)


# ------------------------------------------------------------- attacks
def test_a_child_directory_replaced_after_the_photograph_refuses(tmp_path):
    root = tree(tmp_path)
    c = DC.Custody(root)
    replace_cell(root)
    with pytest.raises(DC.DirectoryIdentityRefusal) as e:
        c.walk_to("cell")
    c.close()
    assert e.value.stage == "OPEN_VS_INVENTORY"
    assert "inode" in e.value.diverged


def test_a_directory_substituted_and_restored_by_name_refuses(tmp_path):
    root = tree(tmp_path)
    c = DC.Custody(root)
    os.rename(root / "cell", root / "cell.original")
    (root / "cell").mkdir()
    os.rename(root / "cell", root / "cell.substitute")
    os.rename(root / "cell.original", root / "cell")
    with pytest.raises(DC.DirectoryIdentityRefusal) as e:
        c.walk_to("cell")
    c.close()
    assert "ctime_ns" in e.value.diverged


def test_an_intermediate_directory_replaced_refuses(tmp_path):
    root = tmp_path / "evidence"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "x.json").write_text("{}")
    c = DC.Custody(root)
    os.rename(root / "a", root / "a.original")
    (root / "a" / "b").mkdir(parents=True)
    with pytest.raises(DC.DirectoryIdentityRefusal):
        c.walk_to("a/b")
    c.close()


def _listing_that_mutates(target_dir: Path):
    real = os.listdir

    def listing(fd):
        names = real(fd)
        (target_dir / "late.json").write_text("{}")
        return names
    return real, listing


def test_a_child_listing_that_changes_during_enumeration_refuses(tmp_path,
                                                                 monkeypatch):
    root = tree(tmp_path)
    c = DC.Custody(root)
    # count descriptors BEFORE patching listdir: fds() itself lists
    # /proc/self/fd, and a patched listdir would mutate the directory
    # before the attack starts
    before = fds()
    real, listing = _listing_that_mutates(root / "cell")
    monkeypatch.setattr(DC.os, "listdir", listing)
    with pytest.raises(DC.DirectoryIdentityRefusal) as e:
        c.walk_to("cell")
    monkeypatch.setattr(DC.os, "listdir", real)
    assert e.value.stage == "ENUMERATION_UNSTABLE"
    assert fds() == before, "the child descriptor must be closed on refusal"
    c.close()


def test_a_root_listing_that_changes_during_enumeration_refuses(tmp_path,
                                                                monkeypatch):
    root = tree(tmp_path)
    real, listing = _listing_that_mutates(root)
    before = fds()
    monkeypatch.setattr(DC.os, "listdir", listing)
    with pytest.raises(DC.DirectoryIdentityRefusal) as e:
        DC.Custody(root)
    monkeypatch.setattr(DC.os, "listdir", real)
    assert e.value.stage == "ENUMERATION_UNSTABLE"
    assert fds() == before


@pytest.mark.parametrize("payload,needle", [
    ('{"rows_used": 999, "rows_used": 11}', "duplicate"),
    ('{"a": {"b": 1, "b": 2}}', "duplicate"),
    ('{"value": NaN}', "NaN"),
    ('{"value": Infinity}', "Infinity"),
    ('{"value": -Infinity}', "-Infinity"),
])
def test_lenient_json_is_refused(tmp_path, payload, needle):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "doc.json").write_text(payload)
    with DC.Custody(root) as c:
        art = c.root_snapshot().read("doc.json")
        with pytest.raises(DC.StrictJsonRefusal) as e:
            art.json()
    assert needle in e.value.reason


def test_descriptors_close_on_success_and_on_exception(tmp_path):
    root = tree(tmp_path)
    before = fds()
    with DC.Custody(root) as c:
        c.walk_to("cell").read("terminal.json")
    assert fds() == before
    c = DC.Custody(root)
    replace_cell(root)
    with pytest.raises(DC.DirectoryIdentityRefusal):
        c.walk_to("cell")
    c.close()
    assert fds() == before


# ------------------------------------------------------------- mutants
def test_mutant_without_directory_binding_consumes_the_replacement(
        tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "check_directory_open_against_inventory",
                        lambda *a, **k: None)
    root = tree(tmp_path)
    with DC.Custody(root) as c:
        replace_cell(root)
        assert c.walk_to("cell").read("terminal.json").json() == {"wall": 9}


@pytest.mark.parametrize("kept", [("after",), ("before",), ()])
def test_mutant_without_either_enumeration_fstat_accepts_a_moving_listing(
        tmp_path, monkeypatch, kept):
    """With the relisting guard off, only the fstats remain; removing
    either one leaves nothing compared."""
    monkeypatch.setattr(DC, "RELIST_ENUMERATION", False)
    monkeypatch.setattr(DC, "ENUMERATION_FSTATS", kept)
    root = tree(tmp_path)
    with DC.Custody(root) as c:
        real, listing = _listing_that_mutates(root / "cell")
        monkeypatch.setattr(DC.os, "listdir", listing)
        snap = c.walk_to("cell")
        monkeypatch.setattr(DC.os, "listdir", real)
        assert "late.json" not in snap.files


def test_mutant_with_lenient_json_accepts_the_duplicate(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(DC, "strict_json_loads",
                        lambda raw, rel: json.loads(raw.decode()))
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "doc.json").write_text('{"k": 1, "k": 2}')
    with DC.Custody(root) as c:
        assert c.root_snapshot().read("doc.json").json() == {"k": 2}


def test_mutant_without_relisting_misses_a_same_tick_listing_change(
        tmp_path, monkeypatch):
    """A directory's timestamps cannot be relied on to move for a change
    made inside one timestamp tick; with relisting off, a mutation whose
    fstats happen to agree is accepted. The fstats are frozen here to
    make that tick deterministic."""
    monkeypatch.setattr(DC, "RELIST_ENUMERATION", False)
    root = tree(tmp_path)
    with DC.Custody(root) as c:
        real_fstat = os.fstat
        frozen = {}

        def fstat(fd):
            st = real_fstat(fd)
            return frozen.setdefault(fd, st)
        real, listing = _listing_that_mutates(root / "cell")
        monkeypatch.setattr(DC.os, "listdir", listing)
        monkeypatch.setattr(DC.os, "fstat", fstat)
        snap = c.walk_to("cell")
        monkeypatch.setattr(DC.os, "listdir", real)
        monkeypatch.setattr(DC.os, "fstat", real_fstat)
        assert "late.json" not in snap.files


def test_the_relisting_guard_names_the_added_entry(tmp_path, monkeypatch):
    root = tree(tmp_path)
    c = DC.Custody(root)
    real, listing = _listing_that_mutates(root / "cell")
    monkeypatch.setattr(DC.os, "listdir", listing)
    with pytest.raises(DC.DirectoryIdentityRefusal) as e:
        c.walk_to("cell")
    monkeypatch.setattr(DC.os, "listdir", real)
    c.close()
    assert e.value.diverged.get("entries", {}).get("added") == ["late.json"] \
        or "mtime_ns" in e.value.diverged
