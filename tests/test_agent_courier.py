"""Courier tests: addressing, dedupe, delivery templating, real-repo
scan against a fixture git repository."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "agent_courier", REPO / "tools" / "agent_courier.py")
courier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(courier)


class TestAddressing:
    @pytest.mark.parametrize("name,identity,expected", [
        ("MUSASHI_TO_GENERAL_SATOSHI_ORDER.md", "satoshi", True),
        ("MUSASHI_TO_GENERAL_SATOSHI_III_ORDER.md", "satoshi", True),
        ("SATOSHI_TO_SERGEANT_RETSU_METHOD.md", "retsu", True),
        ("SATOSHI_TO_GENERAL_MUSASHI_AMENDMENT_1.md", "musashi", True),
        ("MUSASHI_TO_GENERAL_SATOSHI_ORDER.md", "musashi", False),
        ("SATOSHI_RETURN_PACKET_209_220.md", "satoshi", False),
        ("AUDIT_SATOSHI_WP4_RETURN.md", "satoshi", False),
    ])
    def test_addressed_to(self, name, identity, expected):
        assert courier.addressed_to(name, identity) is expected

    def test_multi_hop_document_reaches_each_recipient_once(self):
        name = "MUSASHI_TO_SATOSHI_AND_TO_RETSU_JOINT.md"
        assert courier.addressed_to(name, "satoshi")
        assert courier.addressed_to(name, "retsu")
        assert not courier.addressed_to(name, "musashi")


@pytest.fixture()
def fixture_repo(tmp_path):
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)],
                   check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)],
                   check=True)
    (work / "docs/handoffs").mkdir(parents=True)
    doc = work / "docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_TEST.md"
    doc.write_text("# orden de prueba\n")
    for cmd in (["git", "add", "-A"],
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "-m", "doc"],
                ["git", "push", "-q", "origin", "HEAD:main"]):
        subprocess.run(cmd, cwd=work, check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)],
                   check=True)
    return clone


class TestScanAndDeliver:
    def test_scan_finds_addressed_document(self, fixture_repo):
        docs = courier.scan_repo(fixture_repo, "satoshi", [])
        assert len(docs) == 1
        assert docs[0]["name"].startswith("MUSASHI_TO_GENERAL_SATOSHI")

    def test_scan_ignores_other_recipients(self, fixture_repo):
        assert courier.scan_repo(fixture_repo, "musashi", []) == []

    def test_delivery_injects_prompt_into_local_cli(self, fixture_repo,
                                                    tmp_path):
        docs = courier.scan_repo(fixture_repo, "satoshi", [])
        capture = tmp_path / "captured.json"
        fake = tmp_path / "fake_cli.py"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            f"open({str(capture)!r},'w').write("
            "json.dumps(sys.argv[1:]))\n")
        fake.chmod(0o755)
        record = courier.deliver(
            docs[0], repo=fixture_repo, inbox=tmp_path / "inbox",
            command_template=[sys.executable, str(fake), "{prompt}"],
            dry_run=False)
        assert record["result"] == "DELIVERED"
        args = json.loads(capture.read_text())
        assert "orden de prueba" not in args[-1]   # prompt, not body
        assert "MUSASHI_TO_GENERAL_SATOSHI_TEST.md" in args[-1]
        inbox_copy = Path(record["local_copy"])
        assert inbox_copy.read_text() == "# orden de prueba\n"

    def test_dedupe_is_idempotent_across_runs(self, fixture_repo,
                                              tmp_path):
        state = tmp_path / "state.json"
        argv = ["--identity", "satoshi", "--repo", str(fixture_repo),
                "--once", "--dry-run", "--state", str(state),
                "--inbox", str(tmp_path / "inbox")]
        assert courier.main(argv) == 0
        first = json.loads(state.read_text())["seen_blobs"]
        assert len(first) == 1
        assert courier.main(argv) == 0
        assert json.loads(state.read_text())["seen_blobs"] == first

    def test_dry_run_never_executes_a_cli(self, fixture_repo, tmp_path):
        docs = courier.scan_repo(fixture_repo, "satoshi", [])
        record = courier.deliver(
            docs[0], repo=fixture_repo, inbox=tmp_path / "inbox",
            command_template=["/nonexistent/cli", "{prompt}"],
            dry_run=True)
        assert record["result"] == "DRY_RUN"
