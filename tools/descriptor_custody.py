#!/usr/bin/env python3
"""R11 (order 2026-09-12): read once, from one descriptor, and consume
those bytes.

The B4 closure verified artifacts and then consumed them through
SEPARATE path opens: it parsed the terminal, hashed it afterwards from
a second open, re-opened the per-bar ledger to count rows, and re-read
`status.json` after inventorying it. Every one of those pairs is a
window in which the file can be replaced, so the record certified
bytes it had not consumed. The PRE demonstrates it: a terminal whose
`wall_seconds` reads 999999 on disk is adjudicated as 123.4, with
every descriptor reporting `verified`.

The rule here is narrow and absolute:

    one path resolution -> one descriptor -> one read -> all facts.

A component-by-component `openat` walk with `O_NOFOLLOW` resolves the
path; `fstat` on that same descriptor establishes that it is a regular
file with the expected owner and mode; the bytes are read once; and
the digest, the parse, the row count and every semantic rule are
computed from THOSE bytes. Nothing downstream is handed a path.

A replacement after the read may cause a refusal or may be irrelevant,
but it can no longer change what reached the adjudication.

The pattern is the one `t2_confirmatory.freeze_evidence_file` already
uses for T2 evidence; this is its directory-walking generalization,
written where B4 can reach it.
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
#: is recorded as a declared weakness on every artifact and on the
#: closure, and only the strict mode refuses it. Silently accepting it
#: without saying so would be the dishonest option.
GROUP_WRITE = _stat.S_IWGRP


class CustodyRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"REFUSED: {msg}")


class Artifact:
    """One file, read exactly once, with every fact derived from those
    same bytes.

    The bytes are private. Callers ask for the digest, the parse, the
    line count or the text, and every answer comes from the single
    read — there is no accessor that takes a path.
    """

    __slots__ = ("rel", "_bytes", "sha256", "size", "uid", "mode",
                 "inode", "device", "mtime_ns", "_json",
                 "custody_weakness")

    def __init__(self, rel: str, payload: bytes, st: os.stat_result):
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
        self._json = None
        self.custody_weakness = (
            "GROUP_WRITABLE" if (self.mode & GROUP_WRITE) else "NONE")

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
                "custody_weakness": self.custody_weakness}


class DirSnapshot:
    """The entries of one directory, listed once from one descriptor.

    R12: classification reads THIS, never a fresh `glob`, `exists` or
    `is_file`. A file that appears after the snapshot cannot change the
    run that used it.
    """

    __slots__ = ("rel", "files", "dirs", "others")

    def __init__(self, rel: str, files: tuple, dirs: tuple,
                 others: tuple):
        self.rel = rel
        self.files = files
        self.dirs = dirs
        self.others = others

    def has_file(self, name: str) -> bool:
        return name in self.files

    def matching(self, prefix: str = "", suffix: str = "") -> tuple:
        return tuple(n for n in self.files
                     if n.startswith(prefix) and n.endswith(suffix))

    def facts(self) -> dict:
        return {"rel": self.rel, "files": sorted(self.files),
                "dirs": sorted(self.dirs),
                "other_entries": sorted(self.others)}


class Custody:
    """A retained root descriptor and everything reached through it."""

    def __init__(self, root: Path, *, require_owner: bool = True,
                 strict_mode: bool = False,
                 expected_uid: int | None = None) -> None:
        self.root = Path(root)
        self.require_owner = require_owner
        #: whose files these must be. Defaults to the reading process.
        #: Naming it explicitly lets a battery demand an owner the
        #: evidence does not have, which is the only way to prove the
        #: check fires without creating files as another user.
        self.expected_uid = (os.getuid() if expected_uid is None
                             else int(expected_uid))
        #: strict_mode also refuses group-writable evidence. The
        #: battery uses it; the real campaign root cannot, because its
        #: artifacts were written 0o664 before this rule existed.
        self.strict_mode = strict_mode
        try:
            self._root_fd = os.open(str(self.root),
                                    os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW)
        except FileNotFoundError:
            raise CustodyRefusal(f"{self.root} does not exist")
        except OSError as exc:
            raise CustodyRefusal(
                f"{self.root} is not an openable directory without "
                f"following a link (errno {exc.errno})")
        st = os.fstat(self._root_fd)
        if not _stat.S_ISDIR(st.st_mode):
            os.close(self._root_fd)
            raise CustodyRefusal(f"{self.root} is not a directory")
        self._reads: list[dict] = []

    # ------------------------------------------------------- walking
    def _walk(self, rel: str) -> tuple[int, str]:
        """Open every component with O_NOFOLLOW, returning the fd of
        the PARENT directory and the leaf name.

        Resolving the whole path in one call would let a symlink
        anywhere along it point the read somewhere else.
        """
        parts = [p for p in str(rel).split("/") if p not in ("", ".")]
        if not parts:
            raise CustodyRefusal("an empty relative path names nothing")
        if any(p == ".." for p in parts):
            raise CustodyRefusal(
                f"{rel!r} escapes the retained root; a custody walk "
                "never climbs out of it")
        fd = self._root_fd
        opened: list[int] = []
        try:
            for part in parts[:-1]:
                nfd = os.open(part, os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
                              dir_fd=fd)
                opened.append(nfd)
                fd = nfd
            return fd, parts[-1]
        except OSError as exc:
            for f in opened:
                os.close(f)
            if exc.errno == errno.ELOOP:
                raise CustodyRefusal(
                    f"{rel}: a path component is a symbolic link")
            if exc.errno == errno.ENOENT:
                raise CustodyRefusal(f"{rel}: a path component is absent")
            if exc.errno == errno.ENOTDIR:
                # O_DIRECTORY|O_NOFOLLOW on a symlink reports ENOTDIR,
                # not ELOOP. Say which it actually is: "not a
                # directory" and "is a link pointing elsewhere" are
                # different facts and a reviewer needs the right one.
                raise CustodyRefusal(
                    f"{rel}: a path component is a symbolic link or "
                    "not a directory")
            raise CustodyRefusal(
                f"{rel}: path component unopenable (errno {exc.errno})")
        finally:
            self._pending = opened

    def _release(self) -> None:
        for f in getattr(self, "_pending", ()):  # parents of the leaf
            try:
                os.close(f)
            except OSError:
                pass
        self._pending = []

    # --------------------------------------------------------- files
    def read(self, rel: str) -> Artifact:
        """Read one file ONCE and return every fact about those bytes."""
        parent_fd, leaf = self._walk(rel)
        try:
            try:
                fd = os.open(leaf, os.O_RDONLY | O_NOFOLLOW,
                             dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise CustodyRefusal(
                        f"{rel}: the leaf is a symbolic link")
                if exc.errno == errno.ENOENT:
                    raise CustodyRefusal(f"{rel}: does not exist")
                raise CustodyRefusal(
                    f"{rel}: unopenable (errno {exc.errno})")
            try:
                st = os.fstat(fd)
                if not _stat.S_ISREG(st.st_mode):
                    raise CustodyRefusal(f"{rel}: not a regular file")
                if self.require_owner and st.st_uid != self.expected_uid:
                    raise CustodyRefusal(
                        f"{rel}: owned by uid {st.st_uid}, not by the "
                        f"expected owner {self.expected_uid}")
                mode = _stat.S_IMODE(st.st_mode)
                if mode & WORLD_WRITE:
                    raise CustodyRefusal(
                        f"{rel}: mode {oct(mode)} is world-writable — "
                        "anybody on this host could have replaced it, "
                        "so it is not evidence")
                if self.strict_mode and (mode & GROUP_WRITE):
                    raise CustodyRefusal(
                        f"{rel}: mode {oct(mode)} is group-writable and "
                        "strict custody was requested")
                chunks = []
                while True:
                    block = os.read(fd, 1 << 20)
                    if not block:
                        break
                    chunks.append(block)
                payload = b"".join(chunks)
                # The size fstat reported and the size we read must
                # agree; a file growing under us is a replacement.
                if st.st_size != len(payload):
                    raise CustodyRefusal(
                        f"{rel}: fstat said {st.st_size} bytes and "
                        f"{len(payload)} were read — the file changed "
                        "while it was being read")
            finally:
                os.close(fd)
        finally:
            self._release()
        art = Artifact(str(rel), payload, st)
        self._reads.append(art.facts())
        return art

    # ---------------------------------------------------- directories
    def snapshot(self, rel: str = "") -> DirSnapshot:
        """List one directory ONCE, classifying entries from that
        listing's own stat, taken through the directory's descriptor."""
        if rel in ("", "."):
            parent_fd, leaf, close_leaf = None, None, False
            dir_fd = self._root_fd
        else:
            parent_fd, leaf = self._walk(rel)
            try:
                dir_fd = os.open(leaf,
                                 os.O_RDONLY | O_DIRECTORY | O_NOFOLLOW,
                                 dir_fd=parent_fd)
            except OSError as exc:
                self._release()
                if exc.errno == errno.ENOENT:
                    raise CustodyRefusal(f"{rel}: directory absent")
                if exc.errno in (errno.ENOTDIR, errno.ELOOP):
                    raise CustodyRefusal(
                        f"{rel}: a symbolic link or not a directory")
                raise CustodyRefusal(
                    f"{rel}: directory unopenable (errno {exc.errno})")
            close_leaf = True
        try:
            names = sorted(os.listdir(dir_fd))
            files, dirs, others = [], [], []
            for name in names:
                try:
                    st = os.stat(name, dir_fd=dir_fd,
                                 follow_symlinks=False)
                except OSError:
                    others.append(name)
                    continue
                if _stat.S_ISREG(st.st_mode):
                    files.append(name)
                elif _stat.S_ISDIR(st.st_mode):
                    dirs.append(name)
                else:
                    others.append(name)
        finally:
            if close_leaf:
                os.close(dir_fd)
            self._release()
        return DirSnapshot(str(rel), tuple(files), tuple(dirs),
                           tuple(others))

    def exists_in(self, snapshot: DirSnapshot, name: str) -> bool:
        """Existence answered FROM a snapshot, never from the
        filesystem. R12: a bare `exists()` at classification time is
        how a cell that appears afterwards keeps NOT_STARTED."""
        return (name in snapshot.files or name in snapshot.dirs
                or name in snapshot.others)

    # -------------------------------------------------------- ledger
    def reads(self) -> list[dict]:
        """Every read this custody performed, in order. The closure
        publishes it so a reviewer can count the opens."""
        return list(self._reads)

    def close(self) -> None:
        try:
            os.close(self._root_fd)
        except OSError:
            pass

    def __enter__(self) -> "Custody":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
