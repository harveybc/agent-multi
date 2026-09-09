"""POST for order T2 C82-C88: the frozen-evidence consumption
closes the PRE counterexample, and the mutations BITE.

Phase 1 (corrected code, in-process): the PRE's exact adversary —
gate design A, replace the active pathname with design B before
the executor continues — now consumes NOTHING: the plan census
shows A's 242 units and the authority names A. Zero writes.

Phase 2a (mutant A, subprocess): reintroducing the post-gate
reread ALONE still refuses — the C86 final-use re-diff catches
design B (defense in depth: two independent layers).

Phase 2b (mutant A2, subprocess): reintroducing the reread AND
deleting the C86 re-diff reopens the PRE exactly — the swap
adversary consumes B again (plan census = 3, authority names A).
The single-snapshot rule is load-bearing.

Phase 3 (mutant B, subprocess): deleting ONLY the C86 final-
point-of-use re-diff lets a snapshot tampered after the gate (one
scientific delta, repaired self digest) reach the plan without
refusal — the corrected code refuses `SCIENTIFIC delta`. The
final re-diff is load-bearing.

All worlds are tmp COPIES of successor/manifest/census; the real
STATE files are never modified. Zero sealed-bank scoring, zero
results roots, no external record authored. CPU only."""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MODE = os.environ.get("T2_C82_POST_CHILD")


def build_world(tmp, conf, ex):
    S = Path.home() / ".local/share/agent-multi"
    succ = tmp / "successor.json"
    man = tmp / "manifest.json"
    cen = tmp / "census.json"
    succ.write_bytes(
        (S / "t2_screen_design_RESOURCE_SUCCESSOR_V1.json"
         ).read_bytes())
    man.write_bytes(
        (S / "t2_public_data_manifest_20260906.json").read_bytes())
    cen.write_bytes(
        (S / "t2_bank_census_20260906.json").read_bytes())
    for f in (succ, man, cen):
        os.chmod(f, 0o600)
    ex.SUCCESSOR_PATH = succ
    ex.MANIFEST_PATH = man
    ex.CENSUS_PATH = cen
    ra = tmp / "auth" / "agent-multi" / "reviewer_authority"
    for d in (tmp / "auth", tmp / "auth" / "agent-multi", ra):
        d.mkdir(mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
    d = json.loads(succ.read_text())
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                           "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    tree = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                           "HEAD^{tree}"], capture_output=True,
                          text=True).stdout.strip()
    rec = {"schema": "agent_multi.musashi_t2_successor_"
                     "execution_record.v1",
           "reviewed_at_date": "2026-09-08",
           "reviewer": "General Musashi",
           "decision": "OPEN_T2_SUCCESSOR_EXECUTION",
           "successor_design_file_sha256":
               hashlib.sha256(succ.read_bytes()).hexdigest(),
           "successor_design_self_sha256": d["design_sha256"],
           "design_review_record_sha256":
               conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
           "manifest_sha256":
               hashlib.sha256(man.read_bytes()).hexdigest(),
           "census_sha256":
               hashlib.sha256(cen.read_bytes()).hexdigest(),
           "executor_code_identity": conf.executor_code_identity(),
           "pinned_commit": head, "pinned_tree": tree}
    er = ra / "MUSASHI_T2_SUCCESSOR_EXECUTION_RECORD.json"
    er.write_text(json.dumps(rec))
    os.chmod(er, 0o600)
    conf.T2_SUCCESSOR_EXECUTION_RECORD_PATH = er
    conf.verify_executor_checkout = \
        lambda c, t, repo_root=None: None
    return succ


def design_b_text(succ_text):
    b = json.loads(succ_text)
    keep = b["task_population"]["series_ids"][:3]
    b["task_population"]["series_ids"] = keep
    b["task_population"]["unit_map"] = {
        k: b["task_population"]["unit_map"][k] for k in keep}
    b["resource_contract"]["max_wall_seconds"] = 999999
    body = {k: b[k] for k in sorted(b) if k != "design_sha256"}
    b["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    return json.dumps(b, indent=1)


def run_adversary(conf, ex, succ, adversary):
    real = conf.verify_confirmatory_gates
    a_sha = hashlib.sha256(succ.read_bytes()).hexdigest()
    bt = design_b_text(succ.read_text())

    def wrapped(*a, **k):
        facts = real(*a, **k)
        if adversary == "swap":
            succ.unlink()
            succ.write_text(bt)
            os.chmod(succ, 0o600)
        elif adversary == "tamper":
            doc = facts["design"].doc
            doc["practical_margin_mase"] = 0.001
            body = {k2: doc[k2] for k2 in sorted(doc)
                    if k2 != "design_sha256"}
            doc["design_sha256"] = hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest()
        return facts
    conf.verify_confirmatory_gates = wrapped
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = ex.main(["--plan"])
        plan = json.loads(buf.getvalue())
        return {"rc": rc, "refused": None,
                "plan_units": plan["plan"]["units"],
                "authority_names_A": plan["authority"]
                ["sealed_design_file_sha256"] == a_sha[:16]}
    except SystemExit as exc:
        return {"rc": "refused", "refused": str(exc)[:90],
                "plan_units": None, "authority_names_A": None}
    finally:
        conf.verify_confirmatory_gates = real


if MODE:
    # child: mutated tools dir FIRST on the path
    tools_dir = Path(os.environ["T2_C82_TOOLS_DIR"])
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tools"))
    sys.path.insert(0, str(tools_dir))
    import t2_confirmatory as conf  # noqa: E402
    import t2_confirmatory_executor as ex  # noqa: E402
    assert Path(ex.__file__).parent == tools_dir, ex.__file__
    tmp = Path(tempfile.mkdtemp(prefix="t2_c82_child_"))
    try:
        succ = build_world(tmp, conf, ex)
        print(json.dumps(run_adversary(conf, ex, succ, MODE)))
    finally:
        shutil.rmtree(tmp)
    sys.exit(0)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c82_post_"))
try:
    # ---- Phase 1: corrected code kills the PRE adversary ----
    succ = build_world(TMP / "w1", conf, ex) if (
        (TMP / "w1").mkdir() or True) else None
    r1 = run_adversary(conf, ex, succ, "swap")
    print("phase1_corrected_swap:", json.dumps(r1))
    assert r1["rc"] == 0 and r1["plan_units"] == 242
    assert r1["authority_names_A"] is True

    (TMP / "w1b").mkdir()
    succ = build_world(TMP / "w1b", conf, ex)
    r1b = run_adversary(conf, ex, succ, "tamper")
    print("phase1_corrected_tamper:", json.dumps(r1b))
    assert r1b["rc"] == "refused"
    assert "SCIENTIFIC delta" in r1b["refused"]

    # ---- Phase 2a: mutant A — post-gate reread restored ----
    src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    needle = '    design = facts["design"].doc\n' \
             '    manifest = facts["manifest"].doc\n'
    assert needle in src
    reread = ('    design = conf.strict_json_load(\n'
              '        active_design_path(), "active design")\n'
              '    manifest = conf.strict_json_load(\n'
              '        MANIFEST_PATH, "manifest")\n')
    diff_needle = ('    if design.get("schema") == '
                   'conf.T2_SUCCESSOR_SCHEMA:\n'
                   '        conf.verify_resource_successor('
                   'design)\n')
    assert diff_needle in src
    mut_a = src.replace(needle, reread)
    da = TMP / "mut_a"
    da.mkdir()
    (da / "t2_confirmatory_executor.py").write_text(mut_a)
    env = {**os.environ, "T2_C82_POST_CHILD": "swap",
           "T2_C82_TOOLS_DIR": str(da)}
    rc = subprocess.run([sys.executable, __file__],
                        capture_output=True, text=True, env=env)
    r2 = json.loads(rc.stdout.strip().splitlines()[-1])
    print("phase2a_mutantA_reread_alone:", json.dumps(r2))
    # defense in depth: the C86 re-diff still refuses B
    assert r2["rc"] == "refused"

    # ---- Phase 2b: mutant A2 — reread AND no final re-diff ----
    mut_a2 = src.replace(needle, reread).replace(diff_needle, "")
    da2 = TMP / "mut_a2"
    da2.mkdir()
    (da2 / "t2_confirmatory_executor.py").write_text(mut_a2)
    env = {**os.environ, "T2_C82_POST_CHILD": "swap",
           "T2_C82_TOOLS_DIR": str(da2)}
    rc = subprocess.run([sys.executable, __file__],
                        capture_output=True, text=True, env=env)
    r2b = json.loads(rc.stdout.strip().splitlines()[-1])
    print("phase2b_mutantA2_pre_reopens:", json.dumps(r2b))
    assert r2b["plan_units"] == 3, "mutant A2 must consume B"
    assert r2b["authority_names_A"] is True  # the split is back

    # ---- Phase 3: mutant B — C86 final re-diff removed ----
    mut_b = src.replace(diff_needle, "")
    db = TMP / "mut_b"
    db.mkdir()
    (db / "t2_confirmatory_executor.py").write_text(mut_b)
    env = {**os.environ, "T2_C82_POST_CHILD": "tamper",
           "T2_C82_TOOLS_DIR": str(db)}
    rc = subprocess.run([sys.executable, __file__],
                        capture_output=True, text=True, env=env)
    r3 = json.loads(rc.stdout.strip().splitlines()[-1])
    print("phase3_mutantB_no_final_diff:", json.dumps(r3))
    assert r3["rc"] == 0, "mutant B must accept the tampered doc"

    print("\nPOST CONFIRMED: the frozen snapshot kills the "
          "post-gate swap (242/A vs the PRE's 3/B), the C86 "
          "final-use diff kills the tampered snapshot, and BOTH "
          "protections bite under mutation")
finally:
    shutil.rmtree(TMP)
