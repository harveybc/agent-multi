"""AUD-SEC-20260810-215: the pre-push sensitivity gate blocks outgoing
commits that carry credentials, account identifiers, signer/key paths,
third-party content markers, live SL/TP levels or topology beyond the
allowlist — with TYPED findings, REDACTED excerpts and nonzero exit.

All fixtures are synthetic values in scratch git repositories; nothing
here is a real credential, account or host.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tools.prepush_sensitivity_gate as gate  # noqa: E402


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   capture_output=True, text=True)


def _sha(repo, ref="HEAD"):
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", ref], check=True,
        capture_output=True, text=True).stdout.strip()


def _repo(tmp_path):
    repo = tmp_path / "scratch-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate-test@example.invalid")
    _git(repo, "config", "user.name", "Gate Test")
    (repo / "README.md").write_text("clean baseline\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "baseline")
    return repo, _sha(repo)


def _commit(repo, relpath, content, message="change"):
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    _git(repo, "add", relpath)
    _git(repo, "commit", "-q", "-m", message)
    return _sha(repo)


def _allowlist(tmp_path, **overrides):
    payload = {
        "schema": "agent_multi.prepush_sensitivity_allowlist.v1",
        "allowed_hostnames": ["localhost", "omega"],
        "allowed_ips": ["127.0.0.1"],
        "allowed_path_globs": {},
    }
    payload.update(overrides)
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(payload))
    return path


def _scan(tmp_path, repo, base, **overrides):
    allowlist = gate.load_allowlist(repo, _allowlist(tmp_path, **overrides))
    commits = [c for c in subprocess.run(
        ["git", "-C", str(repo), "rev-list", f"{base}..HEAD"],
        check=True, capture_output=True, text=True).stdout.split()]
    return gate.scan_commits(repo, commits, allowlist)


# ── one fixture per sensitivity class ──

def test_credentials_are_found_and_redacted(tmp_path):
    repo, base = _repo(tmp_path)
    secret = "hunter2hunter2hunter2"
    _commit(repo, "notes.txt",
            f'api_key = "{secret}"\nAKIAABCDEFGHIJKLMNOP\n')
    findings = _scan(tmp_path, repo, base)
    classes = {f["class"] for f in findings}
    rules = {f["rule"] for f in findings}
    assert classes == {"credentials"}
    assert {"secret_assignment", "aws_access_key_id"} <= rules
    dumped = json.dumps(findings)
    # typed, bound to commit/path/line — and the value itself NEVER echoes
    assert secret not in dumped
    assert "AKIAABCDEFGHIJKLMNOP" not in dumped
    assert all(f["path"] == "notes.txt" and f["commit"] for f in findings)
    assert any("[REDACTED:credentials]" in f["excerpt"] for f in findings)


def test_account_identifiers_are_found(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "docs/venue_notes.md",
            "account_number: 4815162342\nbroker code DU7654321\n")
    findings = _scan(tmp_path, repo, base)
    assert {f["class"] for f in findings} == {"account_identifier"}
    assert {f["rule"] for f in findings} == {"numeric_account_assignment",
                                             "ibkr_account_code"}
    assert "4815162342" not in json.dumps(findings)


def test_signer_key_paths_are_found(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "deploy.txt",
            "signer: /home/operator/.ssh/id_ed25519\n"
            "backup at /srv/keys/owner-signer.pem\n")
    findings = _scan(tmp_path, repo, base)
    assert {f["class"] for f in findings} == {"signer_key_path"}
    assert len(findings) >= 2


def test_third_party_content_markers_scoped_to_docs(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "docs/audits/evidence/packet.md",
            "author_handle: someone\npost_text: their words\n")
    _commit(repo, "tools/social_tool.py",
            "ROW = {'author_handle': None, 'post_text': None}\n")
    findings = _scan(tmp_path, repo, base)
    assert {f["class"] for f in findings} == {"third_party_content"}
    paths = {f["path"] for f in findings}
    # markers in code stay legal; packets under docs/ are the leak channel
    assert paths == {"docs/audits/evidence/packet.md"}


def test_sl_tp_live_levels_scoped_to_docs(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "docs/audits/evidence/venue_facts.json",
            '{"stop_loss": 3567.25, "take_profit": 3721.5}\n')
    _commit(repo, "examples/config/genome.json",
            '{"stop_loss": 0.05}\n')
    findings = _scan(tmp_path, repo, base)
    assert {f["class"] for f in findings} == {"sl_tp_live_levels"}
    assert {f["path"] for f in findings} == {
        "docs/audits/evidence/venue_facts.json"}
    assert "3567.25" not in json.dumps(findings)


def test_topology_beyond_allowlist_is_found(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "docs/fleet.md",
            'hostname: omega\nhostname: sigma-new\n'
            'peer 127.0.0.1 and peer 192.168.7.42\n'
            'GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326\n')
    findings = _scan(tmp_path, repo, base)
    assert {f["class"] for f in findings} == {"topology"}
    rules = sorted(f["rule"] for f in findings)
    # omega and 127.0.0.1 pass the allowlist; the others are findings
    assert rules == ["gpu_uuid", "hostname_assignment", "ip_address"]
    dumped = json.dumps(findings)
    assert "sigma-new" not in dumped and "192.168.7.42" not in dumped


def test_allowlisted_path_glob_exempts_only_named_class(tmp_path):
    repo, base = _repo(tmp_path)
    _commit(repo, "examples/config/contract.json",
            '{"hostname": "sigma-new", "api_key": "hunter2hunter2h"}\n')
    findings = _scan(
        tmp_path, repo, base,
        allowed_path_globs={"topology": ["examples/config/*"]})
    # topology is exempt there — credentials still fire
    assert {f["class"] for f in findings} == {"credentials"}


# ── CLI contract: exit codes, JSON report, stdin-refs mode ──

def test_clean_range_exits_zero(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    _commit(repo, "docs/clean.md", "an unremarkable document\n")
    rc = gate.main(["--repo", str(repo), "--range", f"{base}..HEAD",
                    "--allowlist", str(_allowlist(tmp_path)), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["outcome"] == "CLEAN"
    assert report["findings"] == []
    assert len(report["scanned_commits"]) == 1


def test_findings_exit_nonzero_with_typed_json_report(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    sha = _commit(repo, "leak.txt", 'password = "correcthorsebattery"\n')
    rc = gate.main(["--repo", str(repo), "--range", f"{base}..HEAD",
                    "--allowlist", str(_allowlist(tmp_path)), "--json"])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert rc == 1
    assert report["outcome"] == "BLOCK"
    finding = report["findings"][0]
    assert finding["class"] == "credentials"
    assert finding["commit"] == sha
    assert finding["path"] == "leak.txt"
    assert finding["line"] == 1
    assert "correcthorsebattery" not in captured.out


def test_stdin_refs_mode_scans_exact_outgoing_range(tmp_path, monkeypatch,
                                                    capsys):
    repo, base = _repo(tmp_path)
    _commit(repo, "leak.txt", "token DU7654321 outgoing\n")
    head = _sha(repo)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"refs/heads/main {head} refs/heads/main {base}\n"))
    rc = gate.main(["--repo", str(repo), "--stdin-refs",
                    "--allowlist", str(_allowlist(tmp_path)), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report["scanned_commits"] == [head]
    assert report["findings"][0]["class"] == "account_identifier"


def test_stdin_refs_deletion_push_scans_nothing(tmp_path, monkeypatch,
                                                capsys):
    repo, base = _repo(tmp_path)
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(f"refs/heads/gone {'0' * 40} refs/heads/gone {base}\n"))
    rc = gate.main(["--repo", str(repo), "--stdin-refs",
                    "--allowlist", str(_allowlist(tmp_path)), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert report["scanned_commits"] == []


def test_truncated_scan_fails_closed(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    for index in range(3):
        _commit(repo, "docs/clean.md", f"revision {index}\n",
                message=f"rev {index}")
    rc = gate.main(["--repo", str(repo), "--range", f"{base}..HEAD",
                    "--allowlist", str(_allowlist(tmp_path)),
                    "--max-commits", "2", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert report["truncated_scan"] is True
    assert report["outcome"] == "BLOCK"


def test_hook_template_execs_the_gate():
    hook = Path(__file__).resolve().parents[2] / "tools/hooks/pre-push"
    text = hook.read_text()
    assert "prepush_sensitivity_gate.py" in text
    assert "--stdin-refs" in text
    assert text.startswith("#!/bin/sh")
