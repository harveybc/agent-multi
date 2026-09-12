"""B4-R22 / C68 / T2-R17: the leaf a read consumes is the leaf the
inventory photographed.

This file is byte-identical in predictor, B4 and T2, and it runs
against each repository's own `tools/descriptor_custody.py`. The last
test pins that module's digest to `tools/descriptor_custody.sha256`, so
three copies cannot drift apart silently.

Every attack keeps what an unprivileged writer can keep equal — name,
mode, length, mtime, and for in-place writes the inode — and changes the
bytes. Each mutant removes one comparison and shows the matching attack
getting through, so none of the guards passes vacuously.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import descriptor_custody as DC  # noqa: E402

A = b'{"wall_seconds": 1.0}'
B = b'{"wall_seconds": 9.0}'


def cell(tmp_path: Path, payload: bytes = A) -> tuple[Path, Path]:
    root = tmp_path / "evidence"
    (root / "cell").mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(root / "cell", 0o700)
    leaf = root / "cell" / "terminal.json"
    leaf.write_bytes(payload)
    os.chmod(leaf, 0o600)
    return root, leaf


def substitute(leaf: Path, payload: bytes, keep_from: os.stat_result):
    """Rename the leaf away and write a replacement with the same name,
    mode, length and mtime."""
    os.rename(leaf, leaf.with_suffix(".orig"))
    leaf.write_bytes(payload)
    os.chmod(leaf, 0o600)
    os.utime(leaf, ns=(keep_from.st_atime_ns, keep_from.st_mtime_ns))


def overwrite_in_place(leaf: Path, payload: bytes,
                       keep_from: os.stat_result):
    with open(leaf, "r+b") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.utime(leaf, ns=(keep_from.st_atime_ns, keep_from.st_mtime_ns))


def photograph(root: Path):
    c = DC.Custody(root)
    return c, c.root_snapshot().subdir("cell")


# ------------------------------------------------------------ positive
def test_an_untouched_leaf_is_read_and_its_binding_published(tmp_path):
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    art = snap.read("terminal.json")
    c.close()
    assert art.json()["wall_seconds"] == 1.0
    b = art.facts()["leaf_binding"]
    assert b["contract"] == DC.CONTRACT
    assert b["inventoried"] == b["fstat_open"] == b["fstat_after_read"]
    assert set(b["inventoried"]) == set(DC.LEAF_IDENTITY_FIELDS)
    assert c.reads()[0]["leaf_binding"] == b


# ------------------------------------------------------------- attacks
def test_a_leaf_replaced_after_inventory_refuses(tmp_path):
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    substitute(leaf, B, os.stat(leaf))
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read("terminal.json")
    c.close()
    assert e.value.stage == "OPEN_VS_INVENTORY"
    assert "inode" in e.value.diverged


def test_same_mode_and_mtime_but_another_inode_refuses(tmp_path):
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    st0 = os.stat(leaf)
    substitute(leaf, B, st0)
    st1 = os.stat(leaf)
    assert (st0.st_mode, st0.st_size, st0.st_mtime_ns) == \
        (st1.st_mode, st1.st_size, st1.st_mtime_ns)
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read("terminal.json")
    c.close()
    assert set(e.value.diverged) >= {"inode"}


def test_a_leaf_replaced_and_restored_by_name_refuses(tmp_path):
    """The original bytes are back under the original name, but nothing
    can prove what the name held in between."""
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    st0 = os.stat(leaf)
    moved = leaf.with_suffix(".orig")
    os.rename(leaf, moved)
    leaf.write_bytes(B)
    os.rename(leaf, leaf.with_suffix(".sub"))
    os.rename(moved, leaf)
    os.utime(leaf, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    assert leaf.read_bytes() == A and os.stat(leaf).st_ino == st0.st_ino
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read("terminal.json")
    c.close()
    assert "ctime_ns" in e.value.diverged


def test_an_equal_length_in_place_mutation_refuses(tmp_path):
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    st0 = os.stat(leaf)
    overwrite_in_place(leaf, B, st0)
    st1 = os.stat(leaf)
    assert (st0.st_ino, st0.st_size, st0.st_mtime_ns) == \
        (st1.st_ino, st1.st_size, st1.st_mtime_ns)
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read("terminal.json")
    c.close()
    assert e.value.diverged == {
        "ctime_ns": {"expected": st0.st_ctime_ns,
                     "actual": st1.st_ctime_ns}}


def _mutate_between_blocks(monkeypatch, leaf: Path, size: int):
    real_read = DC.os.read
    state = {"blocks": 0}

    def reading(fd, n):
        block = real_read(fd, n)
        state["blocks"] += 1
        if state["blocks"] == 1:
            st = os.stat(leaf)
            with open(leaf, "r+b") as fh:
                fh.seek(size - 4)
                fh.write(b"ZZZZ")
            os.utime(leaf, ns=(st.st_atime_ns, st.st_mtime_ns))
        return block

    monkeypatch.setattr(DC, "READ_BLOCK", 4096)
    monkeypatch.setattr(DC.os, "read", reading)
    return state


def test_a_mutation_during_a_block_read_refuses(tmp_path, monkeypatch):
    size = 3 * 4096
    root, leaf = cell(tmp_path, b"x" * size)
    c, snap = photograph(root)
    state = _mutate_between_blocks(monkeypatch, leaf, size)
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap.read("terminal.json")
    monkeypatch.undo()
    c.close()
    assert state["blocks"] >= 2
    assert e.value.stage == "AFTER_READ_VS_OPEN"


def test_no_descriptor_leaks_when_a_leaf_refuses(tmp_path):
    root, leaf = cell(tmp_path)
    before = len(os.listdir("/proc/self/fd"))
    c, snap = photograph(root)
    substitute(leaf, B, os.stat(leaf))
    with pytest.raises(DC.LeafIdentityRefusal):
        snap.read("terminal.json")
    assert len(os.listdir("/proc/self/fd")) == before + 2
    c.close()
    assert len(os.listdir("/proc/self/fd")) == before


def test_a_refusal_returns_no_bytes_and_records_no_read(tmp_path):
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    overwrite_in_place(leaf, B, os.stat(leaf))
    with pytest.raises(DC.LeafIdentityRefusal):
        snap.read("terminal.json")
    assert c.reads() == []
    c.close()


# ------------------------------------------------------------- mutants
def test_mutant_without_the_open_check_consumes_the_substitute(
        tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "check_open_against_inventory",
                        lambda *a, **k: None)
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    substitute(leaf, B, os.stat(leaf))
    assert snap.read("terminal.json").json()["wall_seconds"] == 9.0
    c.close()


def test_mutant_without_the_after_read_check_admits_a_torn_read(
        tmp_path, monkeypatch):
    monkeypatch.setattr(DC, "check_after_read_against_open",
                        lambda *a, **k: None)
    size = 3 * 4096
    root, leaf = cell(tmp_path, b"x" * size)
    c, snap = photograph(root)
    _mutate_between_blocks(monkeypatch, leaf, size)
    art = snap.read("terminal.json")
    monkeypatch.undo()
    c.close()
    assert art.raw().endswith(b"ZZZZ")


@pytest.mark.parametrize("field,attack", [
    ("inode", "substitute"),
    ("size", "grow"),
    ("mode", "chmod"),
    ("mtime_ns", "touch"),
])
def test_mutant_without_one_field_and_ctime_admits_its_attack(
        tmp_path, monkeypatch, field, attack):
    """ctime catches every attack on its own, so each other field is
    shown load-bearing with ctime removed beside it."""
    monkeypatch.setattr(DC, "LEAF_IDENTITY_FIELDS", tuple(
        f for f in DC.LEAF_IDENTITY_FIELDS if f not in (field, "ctime_ns")))
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    st0 = os.stat(leaf)
    if attack == "substitute":
        substitute(leaf, B, st0)
    elif attack == "grow":
        with open(leaf, "ab") as fh:
            fh.write(b" ")
        os.utime(leaf, ns=(st0.st_atime_ns, st0.st_mtime_ns))
    elif attack == "chmod":
        os.chmod(leaf, 0o400)
    elif attack == "touch":
        os.utime(leaf, ns=(st0.st_atime_ns, st0.st_mtime_ns + 10**9))
    snap.read("terminal.json")
    c.close()
    with DC.Custody(root) as full:
        pass
    monkeypatch.undo()
    root2, leaf2 = cell(tmp_path / "control")
    c2, snap2 = photograph(root2)
    if attack == "substitute":
        substitute(leaf2, B, os.stat(leaf2))
    elif attack == "grow":
        with open(leaf2, "ab") as fh:
            fh.write(b" ")
    elif attack == "chmod":
        os.chmod(leaf2, 0o400)
    elif attack == "touch":
        os.utime(leaf2, ns=(0, os.stat(leaf2).st_mtime_ns + 10**9))
    with pytest.raises(DC.LeafIdentityRefusal) as e:
        snap2.read("terminal.json")
    c2.close()
    assert field in e.value.diverged


def test_mutant_without_ctime_admits_the_in_place_write(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(DC, "LEAF_IDENTITY_FIELDS", tuple(
        f for f in DC.LEAF_IDENTITY_FIELDS if f != "ctime_ns"))
    root, leaf = cell(tmp_path)
    c, snap = photograph(root)
    overwrite_in_place(leaf, B, os.stat(leaf))
    assert snap.read("terminal.json").json()["wall_seconds"] == 9.0
    c.close()


# ------------------------------------------------------ one implementation
def test_this_copy_matches_the_pinned_implementation_digest():
    pinned = (REPO / "tools" / "descriptor_custody.sha256").read_text().split()[0]
    actual = hashlib.sha256(
        (REPO / "tools" / "descriptor_custody.py").read_bytes()).hexdigest()
    assert actual == pinned, (
        "this repository's descriptor_custody.py differs from the pinned "
        "shared implementation")
