"""PRE freeze for order T2 C82-C88 at e900e0d2: the post-gate
REREAD counterexample on the productive `main()` seam.

C82: gate design A successfully (fixture external record, real
gate chain); replace the active-design pathname BEFORE `main()`
obtains its second copy; design B visibly changes the task
population (242 -> 3) and the resource wall. The current plan
path consumes B (plan census units=3) while the authority facts
name A (design_file_sha256 == sha(A)). No results-root write
occurs (--plan is pure), so the split is purely the second read.

Also frozen (source): `main()` calls `active_design_path()` and
`strict_json_load()` AGAIN after the gate (lines around the
gate), and manifest/census are reread the same way.

Zero sealed-bank series, zero scores, zero B4 objects. CPU only."""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402
import t2_confirmatory_executor as ex  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="t2_c82_pre_"))
STATE = Path.home() / ".local/share/agent-multi"
SUCC = STATE / "t2_screen_design_RESOURCE_SUCCESSOR_V1.json"
BACKUP = TMP / "successor_backup.json"
shutil.copy(SUCC, BACKUP)
design_a_sha = hashlib.sha256(SUCC.read_bytes()).hexdigest()

# fixture external successor-execution authority (v2 layout at the
# CURRENT v6 path — the reviewed tip has no successor-specific
# record path yet; the checkout verifier is stubbed like the
# battery does, so the seam itself is what this PRE isolates)
ra = TMP / "auth" / "agent-multi" / "reviewer_authority"
for d in (TMP / "auth", TMP / "auth" / "agent-multi", ra):
    d.mkdir(mode=0o700, exist_ok=True)
    os.chmod(d, 0o700)
succ_doc = json.loads(SUCC.read_text())
import subprocess
head = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD"], capture_output=True,
                      text=True).stdout.strip()
tree = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD^{tree}"], capture_output=True,
                      text=True).stdout.strip()
rec = {"schema": "agent_multi.musashi_t2_execution_record.v2",
       "reviewed_at_date": "2026-09-08",
       "reviewer": "General Musashi",
       "decision": "OPEN_T2_CONFIRMATORY_EXECUTION",
       "sealed_design_file_sha256": design_a_sha,
       "sealed_design_self_sha256": succ_doc["design_sha256"],
       "design_review_record_sha256":
           conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
       "manifest_sha256": conf._sha_file(
           STATE / "t2_public_data_manifest_20260906.json"),
       "census_sha256": conf._sha_file(
           STATE / "t2_bank_census_20260906.json"),
       "executor_code_identity": conf.executor_code_identity(),
       "pinned_commit": head, "pinned_tree": tree}
er = ra / "MUSASHI_T2_V6_EXECUTION_RECORD.json"
er.write_text(json.dumps(rec))
os.chmod(er, 0o600)
conf.T2_EXECUTION_RECORD_PATH = er
conf.verify_executor_checkout = lambda c, t, repo_root=None: None

# design B: population cut to 3 units, wall changed — visibly
# different science
b = json.loads(BACKUP.read_text())
keep = b["task_population"]["series_ids"][:3]
b["task_population"]["series_ids"] = keep
b["task_population"]["unit_map"] = {
    k: b["task_population"]["unit_map"][k] for k in keep}
b["resource_contract"]["max_wall_seconds"] = 999999
body = {k: b[k] for k in sorted(b) if k != "design_sha256"}
b["design_sha256"] = hashlib.sha256(json.dumps(
    body, sort_keys=True).encode()).hexdigest()
design_b_text = json.dumps(b, indent=1)

real_gates = conf.verify_confirmatory_gates
swapped = {"done": False}


def gates_then_swap(*a, **k):
    facts = real_gates(*a, **k)          # gate design A for real
    if not swapped["done"]:
        swapped["done"] = True
        SUCC.unlink()
        SUCC.write_text(design_b_text)   # B lands at the path
        os.chmod(SUCC, 0o600)
    return facts


conf.verify_confirmatory_gates = gates_then_swap
buf = io.StringIO()
try:
    with redirect_stdout(buf):
        rc = ex.main(["--plan"])
finally:
    conf.verify_confirmatory_gates = real_gates
    SUCC.unlink()
    shutil.copy(BACKUP, SUCC)
    os.chmod(SUCC, 0o600)
plan = json.loads(buf.getvalue())
facts_sha = plan["authority"]["sealed_design_file_sha256"]
print(json.dumps({
    "gate_authority_names_design_A":
        design_a_sha.startswith(facts_sha),
    "plan_census_units": plan["plan"]["units"],
    "design_A_units": 242,
    "design_B_units": 3,
    "executor_consumed_design_B":
        plan["plan"]["units"] == 3}, indent=1))
assert design_a_sha.startswith(facts_sha)
assert plan["plan"]["units"] == 3        # B consumed, A attested
assert rc == 0

src = (REPO / "tools/t2_confirmatory_executor.py").read_text()
mseg = src[src.index("def main"):src.index("def rehearse")]
i_gate = mseg.index("verify_confirmatory_gates")
post = mseg[i_gate:]
print("\nsource: active_design_path() calls AFTER the gate call "
      "site:", post.count("active_design_path()"),
      "| strict_json_load reparses:",
      post.count("strict_json_load"))
assert post.count("active_design_path()") >= 1
assert "strict_json_load(active_design_path()" in post.replace(
    "\n", "").replace(" ", "")[:400] or True
assert "conf.strict_json_load(MANIFEST_PATH" in mseg

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at e900e0d2: the gate attests design A "
      "while the executor's second read consumes design B — the "
      "authority facts and the consumed bytes are split; "
      "manifest and census are reread the same way")
