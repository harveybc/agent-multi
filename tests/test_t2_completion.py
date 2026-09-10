"""T2 C90: the completed-campaign refusal battery — every
forgery against the executed 242-unit root dies inside the
fresh reconstruction before any adjudication."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

import t2_confirmatory as conf  # noqa: E402
import t2_completion_reconstruction as rec  # noqa: E402

REAL = rec.DEFAULT_ROOT


def _exec_mod():
    import t2_confirmatory_executor as ex
    return ex


@pytest.fixture(scope="module")
def shared_root(tmp_path_factory):
    """ONE trusted-tmp copy of the ~900 MB campaign root for the
    whole module (ten per-test copies exhausted the tmpfs
    quota); each test restores what it mutated."""
    if not REAL.exists():
        pytest.skip("campaign root absent on this host")
    base = tmp_path_factory.mktemp("t2c90")
    root = base / REAL.name
    shutil.copytree(REAL, root)
    for q in root.rglob("*"):
        os.chmod(q, 0o700 if q.is_dir() else 0o600)
    os.chmod(root, 0o700)
    return base, root


@pytest.fixture()
def world(shared_root, monkeypatch):
    base, root = shared_root
    ex = _exec_mod()
    monkeypatch.setattr(
        ex, "_TRUSTED_ROOT_PARENTS",
        tuple(ex._TRUSTED_ROOT_PARENTS) + (base,))
    # the execution record pins Musashi's launch commit; the
    # battery runs at a LATER tip, so ONLY the checkout-identity
    # comparison is stubbed — every external record, digest and
    # unit verification stays real.
    monkeypatch.setattr(conf, "verify_executor_checkout",
                        lambda c, t, repo_root=None: None)
    # claims pin the campaign's executing commit/tree; present
    # exactly those pins so every unit verifies for its OWN
    # reasons and each mutation dies on ITS needle.
    claim0 = sorted((root / "units").glob("CLAIM_*.json"))[0]
    cdoc = json.loads(claim0.read_text())
    monkeypatch.setattr(
        ex, "_git_head_tree",
        lambda: (cdoc["pinned_commit"], cdoc["pinned_tree"]))
    # the battery tip carries corrected executor bytes, so the
    # physical-surface comparison would refuse EVERY test for
    # the same reason; present the campaign's declared code
    # identity so each mutation dies on ITS OWN needle instead.
    rec0 = json.loads(sorted(
        (root / "units").glob("RECORD_*.json"))[0].read_text())
    monkeypatch.setattr(
        conf, "executor_code_identity",
        lambda repo_root=None: rec0["code_identity"])
    before = {}
    for q in root.rglob("*"):
        if q.is_file():
            before[q.relative_to(root)] = q.stat().st_mtime_ns
    yield root, ex
    # selective restore: changed or new files revert to REAL
    for q in list(root.rglob("*")):
        if not q.is_file():
            continue
        rel = q.relative_to(root)
        if rel not in before:
            q.unlink()
        elif q.stat().st_mtime_ns != before[rel]:
            src = REAL / rel
            shutil.copyfile(src, q)
            os.chmod(q, 0o600)
    for rel in before:
        if not (root / rel).exists():
            shutil.copyfile(REAL / rel, root / rel)
            os.chmod(root / rel, 0o600)


def _uid0(root):
    cn = sorted((root / "units").glob("CLAIM_*.json"))[0]
    doc = json.loads(cn.read_text())
    return doc["unit_id"], cn.name[len("CLAIM_"):-len(".json")]


def _selfsha(doc):
    import hashlib
    body = {k: doc[k] for k in sorted(doc)
            if k != "record_sha256"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()


def _rewrite_record(root, safe, mutate):
    p = root / "units" / f"RECORD_{safe}.json"
    doc = json.loads(p.read_text())
    mutate(doc)
    # repair BOTH self digests (inner assay + outer record)
    if isinstance(doc.get("assay_record"), dict):
        doc["assay_record"]["record_sha256"] = _selfsha(
            doc["assay_record"])
    doc["record_sha256"] = _selfsha(doc)
    p.write_text(json.dumps(doc, indent=1))
    os.chmod(p, 0o600)


def _expect_refusal(root, needle):
    with pytest.raises(SystemExit, match=needle):
        rec.reconstruct(root)


def test_altered_prediction_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    def mut(doc):
        res = doc["assay_record"]["rolling_origins"][
            "origin0"]["results"]
        m = res["X"]["ridge"]
        key = ("mase_primary" if "mase_primary" in m
               else next(k for k, v in m.items()
                         if isinstance(v, (int, float))))
        m[key] = 0.000001
    _rewrite_record(root, safe, mut)
    _expect_refusal(root, "re-derive|mismatch|does not")


def test_swapped_observation_array_refuses(world):
    root, ex = world
    uid0, safe0 = _uid0(root)
    arrays = sorted((root / "units").glob("ARRAYS_*.npz"))
    a0, a1 = arrays[0], arrays[1]
    tmp = a0.read_bytes()
    a0.write_bytes(a1.read_bytes())
    a1.write_bytes(tmp)
    for a in (a0, a1):
        os.chmod(a, 0o600)
    _expect_refusal(root, "physical|slice|observation|"
                          "re-derive|does not")


def test_family_relabel_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    def mut(doc):
        ar = doc["assay_record"]
        ar["family"] = ("hospital"
                        if ar.get("family") != "hospital"
                        else "weather")
        doc["unit_binding"]["family"] = ar["family"]
    _rewrite_record(root, safe, mut)
    _expect_refusal(root, "family|schema|does not|re-derive")


def test_shifted_origin_window_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    def mut(doc):
        w = doc["unit_binding"]["origin_windows"]["origin0"]
        w["train"] = [w["train"][0], w["train"][1] - 1]
        a0 = doc["assay_record"]["rolling_origins"]["origin0"]
        a0["train"] = list(w["train"])
    _rewrite_record(root, safe, mut)
    _expect_refusal(root, "origin|window|does not|re-derive")


def test_omitted_mlp_seed_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    def mut(doc):
        res = doc["assay_record"]["rolling_origins"][
            "origin0"]["results"]
        ml = res["X"]["mlp_small"]
        ml.pop(next(k for k in ml if k.startswith("seed")))
    _rewrite_record(root, safe, mut)
    _expect_refusal(root, "seed|mlp|schema|does not|re-derive")


def test_altered_cost_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    def mut(doc):
        c = doc["assay_record"]["costs_by_phase"]
        def bump(node):
            if isinstance(node, dict):
                for k2 in node:
                    if isinstance(node[k2], (int, float)):
                        node[k2] = float(node[k2]) + 123.456
                        return True
                    if bump(node[k2]):
                        return True
            return False
        assert bump(c), "no numeric cost found"
    _rewrite_record(root, safe, mut)
    _expect_refusal(root, "cost|does not|re-derive")


def test_duplicate_unit_refuses(world):
    root, ex = world
    uid, safe = _uid0(root)
    src = root / "units" / f"RECORD_{safe}.json"
    dup = root / "units" / f"RECORD_{safe}__dup.json"
    shutil.copyfile(src, dup)
    os.chmod(dup, 0o600)
    _expect_refusal(root, "foreign|duplicate|exact|inventory")


def test_foreign_file_refuses(world):
    root, ex = world
    fo = root / "units" / "FOREIGN_object.bin"
    fo.write_bytes(b"alien")
    os.chmod(fo, 0o600)
    _expect_refusal(root, "foreign|exact|inventory|unrecognized")


def test_truncated_wall_ledger_refuses(world):
    root, ex = world
    wp = root / "T2_WALL_LEDGER.jsonl"
    raw = wp.read_bytes()
    wp.write_bytes(raw[:-40])          # torn tail
    os.chmod(wp, 0o600)
    _expect_refusal(root, "TORN_TAIL|torn|malformed")


def test_changed_successor_field_refuses(world, monkeypatch,
                                         tmp_path):
    """A successor with one scientific delta refuses at the
    gates before any unit is touched."""
    root, ex = world
    succ = (Path.home() / ".local/share/agent-multi/"
            "t2_screen_design_RESOURCE_SUCCESSOR_V1.json")
    doc = json.loads(succ.read_text())
    doc["practical_margin_mase"] = 0.000123
    body = {k: doc[k] for k in sorted(doc)
            if k != "design_sha256"}
    doc["design_sha256"] = __import__("hashlib").sha256(
        json.dumps(body, sort_keys=True).encode()).hexdigest()
    forged = tmp_path / "forged_successor.json"
    forged.write_text(json.dumps(doc, indent=1))
    os.chmod(forged, 0o600)
    monkeypatch.setattr(ex, "SUCCESSOR_PATH", forged)
    _expect_refusal(root, "SCIENTIFIC delta|self identity|"
                          "does not")
