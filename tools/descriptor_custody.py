#!/usr/bin/env python3
"""Read once, from one descriptor, and consume those bytes — and keep
the DIRECTORY that was photographed.

R11 (2026-09-11) closed the first half: one path resolution, one
descriptor, one read, all facts. The B4 closure used to parse the
terminal and hash it from a second open, count per-bar rows from a
third, and re-read `status.json` after inventorying it.

R16 (2026-09-12) closes the half that was still open, and the audit's
counterexample is exact. `DirSnapshot` kept only NAMES and closed the
directory descriptor; every later `read()` re-walked the path from the
retained root. So a photographed cell directory could be renamed and a
different directory put in its place, and the read consumed the
replacement:

    snapshot lists cell/terminal.json      (wall_seconds = 1.0)
    mv cell cell_moved; mkdir cell; ...    (wall_seconds = 999.0)
    custody.read("cell/terminal.json")  ->  999.0

A photograph that does not retain its subject is a list of names.

So a `DirSnapshot` now HOLDS the directory's own descriptor together
with its `(device, inode, uid, mode)`, every file is opened relative to
THAT descriptor, and a subdirectory is opened relative to its parent's.
After the photograph nothing is ever resolved by name again. Renaming
the directory, replacing it, or restoring the original name over a new
inode all leave the retained descriptor pointing at the instance that
was inventoried.

Descriptors are closed deterministically, children before parents, on
success and on exception alike.

R19-R20 (2026-09-12) closes the LEAF. Retaining the directory kept its
entries' names honest and nothing else: `read()` opened the name again
relative to the retained descriptor and compared nothing it had seen at
inventory time. So a leaf renamed away inside that same directory and
replaced by a file with the same name, mode, length and mtime was
consumed, and so was an equal-length in-place write that put the mtime
back:

    inventory  terminal.json   wall_seconds = 1.0
    mv terminal.json terminal.orig; write terminal.json (9.0);
    touch -d <same mtime> terminal.json
    cell.read("terminal.json")  ->  9.0

The inventory now keeps, per name, the file's type, device, inode, uid,
mode, size, mtime_ns and ctime_ns. A read is accepted only if

  * the `fstat` of the descriptor it opened equals the inventory on
    every one of those fields, and
  * a second `fstat` taken after the last byte equals the first.

ctime is the field an unprivileged writer cannot put back, which is why
it is compared at both points. Any divergence raises
`LeafIdentityRefusal` naming the stage and the fields, and no byte of
that read leaves this module. The artifact publishes all three fact
sets, so the binding is checkable rather than asserted.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import stat as _stat
from pathlib import Path

O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)

#: world-writable evidence is not evidence: anybody on the host could
#: have replaced it. This ALWAYS refuses.
WORLD_WRITE = _stat.S_IWOTH
#: group-writable is weaker than it should be but is the mode the
#: existing campaign artifacts were actually written with (0o664).
#: Refusing it outright would make the real evidence unreadable, so it
#: is recorded as a declared weakness on every artifact, on every
#: directory and on the closure, and only strict mode refuses it.
#: Silently accepting it without saying so would be the dishonest
#: option; a retroactive `chmod` on the originals would be worse.
GROUP_WRITE = _stat.S_IWGRP

#: The contract every copy of this module implements. predictor, B4 and
#: T2 carry byte-identical copies and one shared fixture file.
CONTRACT = "agent_multi.descriptor_custody.leaf_binding.v1"

#: What the inventory remembers about a leaf, and what both `fstat`
#: comparisons require to be equal.
LEAF_IDENTITY_FIELDS = ("type", "device", "inode", "uid", "mode", "size",
                        "mtime_ns", "ctime_ns")

#: Block size for reading; a module attribute so a battery can mutate a
#: file between blocks.
READ_BLOCK = 1 << 20


class CustodyRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


class LeafIdentityRefusal(CustodyRefusal):
    """The file that was opened, or the file after it was read, is not
    the file the inventory photographed."""

    def __init__(self, rel: str, stage: str, diverged: dict) -> None:
        self.rel = rel
        self.stage = stage
        self.diverged = diverged
        super().__init__(
            f"{rel}: leaf identity diverged at {stage} on "
            f"{sorted(diverged)}; no value derived from these bytes is "
            "returned")


def leaf_facts(st: os.stat_result) -> dict:
    return {"type": _stat.S_IFMT(st.st_mode), "device": st.st_dev,
            "inode": st.st_ino, "uid": st.st_uid,
            "mode": _stat.S_IMODE(st.st_mode), "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def _diverged(expected: dict, actual: dict) -> dict:
    return {f: {"expected": expected[f], "actual": actual[f]}
            for f in LEAF_IDENTITY_FIELDS if expected[f] != actual[f]}


def check_open_against_inventory(rel: str, inventoried: dict,
                                 opened: dict) -> None:
    diff = _diverged(inventoried, opened)
    if diff:
        raise LeafIdentityRefusal(rel, "OPEN_VS_INVENTORY", diff)


def check_after_read_against_open(rel: str, opened: dict,
                                  after: dict) -> None:
    diff = _diverged(opened, after)
    if diff:
        raise LeafIdentityRefusal(rel, "AFTER_READ_VS_OPEN", diff)


def _weakness(mode: int) -> str:
    return "GROUP_WRITABLE" if (mode & GROUP_WRITE) else "NONE"


class Artifact:
    """One file, read exactly once, with every fact derived from those
    same bytes and from the directory instance it was read out of."""

    __slots__ = ("rel", "_bytes", "sha256", "size", "uid", "mode",
                 "inode", "device", "mtime_ns", "_json",
                 "custody_weakness", "dir_inode", "dir_device",
                 "leaf_binding")

    def __init__(self, rel: str, payload: bytes, st: os.stat_result,
                 dir_device: int | None = None,
                 dir_inode: int | None = None,
                 leaf_binding: dict | None = None):
        self.rel = rel
        self._bytes = payload
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.size = len(payload)
        self.uid = st.st_uid
        self.mode = _stat.S_IMODE(st.st_mode)
        self.inode = st.st_ino
        self.device = st.st_dev
        # taken from the SAME fstat as the size and the mode, so a
        # timestamp used as evidence is the timestamp of the bytes read
        self.mtime_ns = st.st_mtime_ns
        # R16: which directory INSTANCE this came out of. A digest that
        # cannot say where it was read from cannot be tied to an
        # inventory.
        self.dir_device = dir_device
        self.dir_inode = dir_inode
        self._json = None
        self.custody_weakness = _weakness(self.mode)
        # R20: the inventoried facts and both fstats this read was
        # accepted under. None only for artifacts built outside a
        # DirSnapshot, which publish that they are unbound.
        self.leaf_binding = leaf_binding

    def json(self) -> dict:
        if self._json is None:
            try:
                self._json = json.loads(self._bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise CustodyRefusal(
                    f"{self.rel}: not parseable as JSON "
                    f"({type(exc).__name__})")
            if not isinstance(self._json, dict):
                raise CustodyRefusal(
                    f"{self.rel}: JSON root is not an object")
        return self._json

    def raw(self) -> bytes:
        """The bytes that were read.

        Deliberately the ONLY way out, and it exists for one purpose:
        handing this same instance to another consumer. It is not a
        re-read and there is no accessor here that takes a path, so a
        second consumer necessarily sees what the first one saw.
        """
        return self._bytes

    def text(self) -> str:
        return self._bytes.decode("utf-8", "replace")

    def lines(self) -> int:
        """Rows in the read bytes, excluding a trailing newline."""
        if not self._bytes:
            return 0
        body = self._bytes
        if body.endswith(b"\n"):
            body = body[:-1]
        return body.count(b"\n") + 1

    def data_rows(self) -> int:
        """CSV rows excluding the header."""
        return max(0, self.lines() - 1)

    def facts(self) -> dict:
        return {"rel": self.rel, "sha256": self.sha256,
                "bytes": self.size, "mode": oct(self.mode),
                "uid": self.uid, "inode": self.inode,
                "device": self.device, "mtime_ns": self.mtime_ns,
                "dir_device": self.dir_device,
                "dir_inode": self.dir_inode,
                "custody_weakness": self.custody_weakness,
                "leaf_binding": (self.leaf_binding if self.leaf_binding
                                 is not None else "UNBOUND")}


class DirSnapshot:
    """One directory, listed once, with its descriptor RETAINED.

    Everything read out of it is opened relative to that descriptor, so
    the bytes consumed provably come from the instance that was
    inventoried — not from whatever now answers to the same name.
    """

    __slots__ = ("rel", "files", "dirs", "others", "device", "inode",
                 "uid", "mode", "custody_weakness", "_fd", "_owner",
                 "_children", "_closed", "_leaf")

    def __init__(self, rel: str, fd: int, st: os.stat_result,
                 owner) -> None:
        self.rel = rel
        self._fd = fd
        self._owner = owner
        self._children: list[DirSnapshot] = []
        self._closed = False
        self.device = st.st_dev
        self.inode = st.st_ino
        self.uid = st.st_uid
        self.mode = _stat.S_IMODE(st.st_mode)
        self.custody_weakness = _weakness(self.mode)
        files, dirs, others = [], [], []
        self._leaf: dict[str, dict] = {}
        for name in sorted(os.listdir(fd)):
            try:
                est = os.stat(name, dir_fd=fd, follow_symlinks=False)
            except OSError:
                others.append(name)
                continue
            if _stat.S_ISREG(est.st_mode):
                files.append(name)
                self._leaf[name] = leaf_facts(est)
            elif _stat.S_ISDIR(est.st_mode):
                dirs.append(name)
            else:
                others.append(name)
        self.files = tuple(files)
        self.dirs = tuple(dirs)
        self.others = tuple(others)

    # ------------------------------------------------------- reading
    def _check_open(self) -> None:
        if self._closed:
            raise CustodyRefusal(
                f"{self.rel or '.'}: this snapshot was closed; a read "
                "after close would have to resolve the path again")

    def read(self, name: str) -> Artifact:
        """Read one file relative to the RETAINED directory descriptor."""
        self._check_open()
        if "/" in name:
            raise CustodyRefusal(
                f"{name!r} is not a name in this directory; read it "
                "from the snapshot of the directory that holds it")
        if name not in self.files:
            raise CustodyRefusal(
                f"{self.rel}/{name}: not in the inventory taken from "
                "this directory instance")
        try:
            fd = os.open(name, os.O_RDONLY | O_NOFOLLOW, dir_fd=self._fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise CustodyRefusal(
                    f"{self.rel}/{name}: the leaf is a symbolic link")
            if exc.errno == errno.ENOENT:
                raise CustodyRefusal(
                    f"{self.rel}/{name}: listed in the inventory but "
                    "gone from the retained directory")
            raise CustodyRefusal(
                f"{self.rel}/{name}: unopenable (errno {exc.errno})")
        rel = f"{self.rel}/{name}" if self.rel else name
        try:
            st = os.fstat(fd)
            opened = leaf_facts(st)
            check_open_against_inventory(rel, self._leaf[name], opened)
            if not _stat.S_ISREG(st.st_mode):
                raise CustodyRefusal(
                    f"{self.rel}/{name}: not a regular file")
            if self._owner.require_owner and \
                    st.st_uid != self._owner.expected_uid:
                raise CustodyRefusal(
                    f"{self.rel}/{name}: owned by uid {st.st_uid}, not "
                    f"by the expected owner {self._owner.expected_uid}")
            mode = _stat.S_IMODE(st.st_mode)
            if mode & WORLD_WRITE:
                raise CustodyRefusal(
                    f"{self.rel}/{name}: mode {oct(mode)} is "
                    "world-writable — anybody on this host could have "
                    "replaced it, so it is not evidence")
            if self._owner.strict_mode and (mode & GROUP_WRITE):
                raise CustodyRefusal(
                    f"{self.rel}/{name}: mode {oct(mode)} is "
                    "group-writable and strict custody was requested")
            chunks = []
            while True:
                block = os.read(fd, READ_BLOCK)
                if not block:
                    break
                chunks.append(block)
            payload = b"".join(chunks)
            after = leaf_facts(os.fstat(fd))
            check_after_read_against_open(rel, opened, after)
            if st.st_size != len(payload):
                raise CustodyRefusal(
                    f"{self.rel}/{name}: fstat said {st.st_size} bytes "
                    f"and {len(payload)} were read — the file changed "
                    "while it was being read")
        finally:
            os.close(fd)
        art = Artifact(rel, payload, st, dir_device=self.device,
                       dir_inode=self.inode,
                       leaf_binding={"contract": CONTRACT,
                                     "inventoried": dict(self._leaf[name]),
                                     "fstat_open": opened,
                                     "fstat_after_read": after})
        self._owner._record(art)
        return art

    def subdir(self, name: str) -> "DirSnapshot":
        """Photograph a child directory, relative to THIS descriptor."""
        self._check_open()
        if "/" in name:
            raise CustodyRefusal(
                f"{name!r} is not a name in this directory")
        if name not in self.dirs:
            raise CustodyRefusal(
                f"{self.rel}/{name}: not a directory in the inventory "
                "taken from this directory instance")
        child = self._owner._open_dir(self._fd,
                                      f"{self.rel}/{name}" if self.rel
                                      else name, name)
        self._children.append(child)
        return child

    # --------------------------------------------------------- facts
    def has_file(self, name: str) -> bool:
        return name in self.files

    def matching(self, prefix: str = "", suffix: str = "") -> tuple:
        return tuple(n for n in self.files
                     if n.startswith(prefix) and n.endswith(suffix))

    def contains(self, name: str) -> bool:
        """Existence answered FROM the inventory, never from the
        filesystem. A bare `exists()` at decision time is how a cell
        that appears afterwards keeps NOT_STARTED."""
        return (name in self.files or name in self.dirs
                or name in self.others)

    def leaf_inventory(self, name: str) -> dict:
        """The facts photographed for `name` at inventory time."""
        return dict(self._leaf[name])

    def facts(self) -> dict:
        return {"rel": self.rel or ".",
                "files": sorted(self.files),
                "dirs": sorted(self.dirs),
                "other_entries": sorted(self.others),
                # R16: the identity of the directory INSTANCE, so a
                # replacement that keeps the name is visible.
                "device": self.device, "inode": self.inode,
                "uid": self.uid, "mode": oct(self.mode),
                "custody_weakness": self.custody_weakness}

    # --------------------------------------------------------- close
    def close(self) -> None:
        if self._closed:
            return
        for child in self._children:
            child.close()
        self._closed = True
        try:
            os.close(self._fd)
        except OSError:
            pass


class Custody:
    """A retained root descriptor, and every directory reached through
    it — all closed deterministically."""

    def __init__(self, root: Path, *, require_owner: bool = True,
                 strict_mode: bool = False,
                 expected_uid: int | None = None) -> None:
        self.root = Path(root)
        self.require_owner = require_owner
        self.strict_mode = strict_mode
        #: whose files these must be. Defaults to the reading process.
        #: Naming it explicitly lets a battery demand an owner the
        #: evidence does not have, which is the only way to prove the
        #: check fires without creating files as another user.
        self.expected_uid = (os.getuid() if expected_uid is None
                             else int(expected_uid))
        self._reads: list[dict] = []
        self._snapshots: list[DirSnapshot] = []
        try:
            fd = os.open(str(self.root),
                         os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        except FileNotFoundError:
            raise CustodyRefusal(f"{self.root} does not exist")
        except OSError as exc:
            raise CustodyRefusal(
                f"{self.root} is not an openable directory without "
                f"following a link (errno {exc.errno})")
        st = os.fstat(fd)
        if not _stat.S_ISDIR(st.st_mode):
            os.close(fd)
            raise CustodyRefusal(f"{self.root} is not a directory")
        if _stat.S_IMODE(st.st_mode) & WORLD_WRITE:
            os.close(fd)
            raise CustodyRefusal(
                f"{self.root}: the evidence root is world-writable")
        self._root = DirSnapshot("", fd, st, self)
        self._snapshots.append(self._root)

    # ---------------------------------------------------- directories
    def _open_dir(self, parent_fd: int, rel: str,
                  name: str) -> DirSnapshot:
        try:
            fd = os.open(name, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
                         dir_fd=parent_fd)
        except OSError as exc:
            if exc.errno in (errno.ENOTDIR, errno.ELOOP):
                raise CustodyRefusal(
                    f"{rel}: a symbolic link or not a directory")
            if exc.errno == errno.ENOENT:
                raise CustodyRefusal(f"{rel}: directory absent")
            raise CustodyRefusal(
                f"{rel}: directory unopenable (errno {exc.errno})")
        try:
            st = os.fstat(fd)
            if not _stat.S_ISDIR(st.st_mode):
                raise CustodyRefusal(f"{rel}: not a directory")
            if self.require_owner and st.st_uid != self.expected_uid:
                raise CustodyRefusal(
                    f"{rel}: directory owned by uid {st.st_uid}, not by "
                    f"the expected owner {self.expected_uid}")
            mode = _stat.S_IMODE(st.st_mode)
            if mode & WORLD_WRITE:
                raise CustodyRefusal(
                    f"{rel}: directory mode {oct(mode)} is "
                    "world-writable; anybody could have replaced its "
                    "contents")
            if self.strict_mode and (mode & GROUP_WRITE):
                raise CustodyRefusal(
                    f"{rel}: directory mode {oct(mode)} is "
                    "group-writable and strict custody was requested")
        except BaseException:
            os.close(fd)
            raise
        snap = DirSnapshot(rel, fd, st, self)
        self._snapshots.append(snap)
        return snap

    def root_snapshot(self) -> DirSnapshot:
        """The retained root. Every other directory is reached from it
        with `subdir`, never by walking a path string again."""
        return self._root

    def walk_to(self, rel: str) -> DirSnapshot:
        """Photograph `rel` by descending from the retained root, one
        RETAINED descriptor per component.

        Every intermediate directory stays open for the life of this
        custody, so no component can be exchanged after it is passed.
        """
        snap = self._root
        parts = [p for p in str(rel).split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise CustodyRefusal(
                f"{rel!r} escapes the retained root; a custody walk "
                "never climbs out of it")
        for part in parts:
            snap = snap.subdir(part)
        return snap

    # --------------------------------------------------------- files
    def read_in(self, dir_rel: str, name: str) -> Artifact:
        """Convenience for a single read: photograph the directory and
        read from that retained instance."""
        return self.walk_to(dir_rel).read(name) if dir_rel \
            else self._root.read(name)

    def _record(self, art: Artifact) -> None:
        self._reads.append(art.facts())

    def reads(self) -> list[dict]:
        """Every read this custody performed, in order. The closure
        publishes it so a reviewer can count the opens."""
        return list(self._reads)

    def open_descriptors(self) -> int:
        return sum(1 for s in self._snapshots if not s._closed)

    # --------------------------------------------------------- bundle
    def export_bundle(self, dest: Path, *, snapshots: list[DirSnapshot],
                      provenance: str) -> dict:
        """R16.4: a strict-mode READ BUNDLE built from the bytes that
        were read through retained descriptors.

        The originals are never touched — not their bytes, not their
        modes. What this produces is a second copy whose custody IS
        strong, carrying the provenance of the weak-mode instance it
        came from, so a later reader can have strict authority without
        anyone having rewritten history to obtain it.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        os.chmod(dest, 0o700)
        written = {}
        for snap in snapshots:
            for name in snap.files:
                art = snap.read(name)
                target = dest / (snap.rel or ".") / name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(target.parent, 0o700)
                fd = os.open(str(target),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o400)
                try:
                    os.write(fd, art.raw())
                    os.fsync(fd)
                finally:
                    os.close(fd)
                written[art.rel] = art.facts()
        doc = {
            "schema": "agent_multi.custody_read_bundle.v1",
            "provenance": provenance,
            "source_root": str(self.root.name),
            "files": written,
            "bundle_sha256": hashlib.sha256(json.dumps(
                written, sort_keys=True).encode()).hexdigest(),
            "originals": "UNTOUCHED — no byte and no mode of the "
                         "source was changed to obtain this bundle",
            "modes": "0o400 files under a 0o700 tree",
        }
        (dest / "BUNDLE.json").write_text(
            json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return doc

    # ---------------------------------------------------------- close
    def close(self) -> None:
        for snap in reversed(self._snapshots):
            snap.close()

    def __enter__(self) -> "Custody":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
