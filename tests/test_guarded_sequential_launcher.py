"""Finding 315 item 7: sequential wrappers refuse identity drift."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "guarded_sequential_launcher",
        REPO / "tools" / "guarded_sequential_launcher.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def repo(tmp_path):
    work = tmp_path / "repo"
    work.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    import os
    (work / "driver.py").write_text("print('x')\n")
    subprocess.run(["git", "add", "-A"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=work,
                   env={**os.environ, **env}, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=work,
                          capture_output=True, text=True).stdout.strip()
    manifest = tmp_path / "identity.json"  # OUTSIDE the repo
    manifest.write_text(json.dumps({
        "full_commit": head,
        "file_sha256": {"driver.py": hashlib.sha256(
            (work / "driver.py").read_bytes()).hexdigest()}}))
    return work, manifest, env


def test_clean_identity_passes(repo):
    tool = _load()
    path, manifest, _ = repo
    out = tool.verify_launch_identity(manifest, path)
    assert out["identity_ok"] and out["files_verified"] == 1


def test_head_drift_refuses(repo):
    import os
    tool = _load()
    path, manifest, env = repo
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m",
                    "docs"], cwd=path, env={**os.environ, **env},
                   check=True)
    with pytest.raises(tool.LaunchIdentityDrift, match="HEAD"):
        tool.verify_launch_identity(manifest, path)


def test_executable_hash_drift_refuses(repo):
    import os
    tool = _load()
    path, manifest, env = repo
    (path / "driver.py").write_text("print('MUTATED')\n")
    subprocess.run(["git", "commit", "-aqm", "mut"], cwd=path,
                   env={**os.environ, **env}, check=True)
    doc = json.loads(manifest.read_text())
    doc["full_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True,
        text=True).stdout.strip()
    manifest.write_text(json.dumps(doc))
    with pytest.raises(tool.LaunchIdentityDrift, match="hash drift"):
        tool.verify_launch_identity(manifest, path)


def test_dirty_tree_refuses(repo):
    tool = _load()
    path, manifest, _ = repo
    (path / "scratch.txt").write_text("dirty")
    with pytest.raises(tool.LaunchIdentityDrift, match="dirty"):
        tool.verify_launch_identity(manifest, path)
