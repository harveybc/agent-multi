"""N3-D4 mutation demonstrations (order @a13671ab §7): each ordered
adversary test BITES — a targeted mutation of the corresponding
guard makes exactly that test fail; green is restored after every
mutation. Counts are read from live pytest output."""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TOOL = REPO / "tools/n3_fresh_confirmation.py"
BAK = Path.home() / ".cache" / "n3f_mutation_bak.py"
PY = str(Path.home()
         / "anaconda3/envs/trading-stack/bin/python")
BATTERY = "tests/unit/test_n3_fresh_confirmation.py"

MUTATIONS = [
    ("M1_anchor_time_guards", """            # adversary 1: a 2025 row used as confirmation
            if dt < conf_lo:
                raise FreshRefusal(
                    f"pre-2026 anchor {t} used as confirmation")
            # adversary 3: future row beyond the sealed interval
            if dt > conf_hi:
                raise FreshRefusal(
                    f"anchor {t} beyond the sealed confirmation "
                    "end")""", """            pass""",
     ["test_adv1_pre2026_anchor", "test_adv3_future_anchor"]),
    ("M2_boundary_guard", """        got = ledger["blocks"].get(name)
        if got is None or _norm(got) != _norm(spec):
            raise FreshRefusal(
                f"role boundary moved: block {name} differs from "
                "the sealed contract")""", """        pass""",
     ["test_adv2_moved_boundary"]),
    ("M3_payload_digest", """        if sha_obj({k: v for k, v in u.items()
                    if k != "payload_sha256"}) != claimed:
            raise FreshRefusal(
                f"unit {u['unit']}: payload altered (digest)")""",
     """        pass""", ["test_adv4_altered_payload"]),
    ("M4_label_history", """        if not all(any(abs(v - c) < 5e-7 for c in candidate)
                   for v in ys_loss[:20]):
            raise FreshRefusal(
                f"unit {u['unit']}: arm1 losses inconsistent with "
                "the bundled fit+cal label histogram — different "
                "label histories")""", """        pass""",
     ["test_adv8_label_history_mismatch"]),
    ("M5_unit_set", """    if len(seen) != len(set(seen)):
        raise FreshRefusal("duplicate units")
    if set(seen) != expected_units:
        raise FreshRefusal(
            f"missing/extra units: {sorted(set(seen) ^ expected_units)[:4]}")""",
     """    seen = sorted(set(seen) & expected_units)""",
     ["test_adv9_missing_unit", "test_duplicate_unit"]),
    ("M6_license_guard", """        if u.get("license_failure"):
            raise FreshRefusal(
                f"unit {u['unit']}: license failure "
                f"{u['license_failure']} beside the decision")""",
     """        pass""",
     ["test_adv9b_license_failure_beside_decision"]),
    ("M7_report_rederivation", """    if verdict != bundle["verdict"]:
        raise FreshRefusal(
            f"report edited: rederived verdict {verdict} != "
            f"bundled {bundle['verdict']}")""", """    pass""",
     ["test_adv10_edited_verdict"]),
    ("M8_float32_coercion", """            if float(r[j]) != float(lake_2025[col].iloc[i]):""",
     """            import numpy as _np32
            if _np32.float32(float(r[j])) != _np32.float32(
                    float(lake_2025[col].iloc[i])):""",
     ["test_sub_float32_revision_refuses"]),
]


def run_battery():
    proc = subprocess.run(
        [PY, "-m", "pytest", BATTERY, "-q"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin",
             "CUDA_VISIBLE_DEVICES": "",
             "HOME": str(Path.home())})
    tail = proc.stdout.strip().splitlines()[-1]
    return tail


shutil.copy(TOOL, BAK)
src_orig = TOOL.read_text()
print("BASELINE:", run_battery())
all_bit = True
for name, old, new, expected_fail in MUTATIONS:
    src = src_orig
    if old not in src:
        print(f"{name}: MUTATION TARGET NOT FOUND — abort")
        all_bit = False
        break
    TOOL.write_text(src.replace(old, new))
    tail = run_battery()
    proc = subprocess.run(
        [PY, "-m", "pytest", BATTERY, "-q"],
        capture_output=True, text=True, cwd=REPO,
        env={"PATH": "/usr/bin:/bin",
             "CUDA_VISIBLE_DEVICES": "",
             "HOME": str(Path.home())})
    failed = [t for t in expected_fail if
              f"{t}" in proc.stdout and "FAILED" in proc.stdout]
    bites = all(any(t in line and "FAILED" in line
                    for line in proc.stdout.splitlines())
                for t in expected_fail)
    print(f"{name}: {tail}  bites={bites} "
          f"(expected failing: {expected_fail})")
    all_bit &= bites
    TOOL.write_text(src_orig)
print("RESTORED:", run_battery())
print("ALL MUTATIONS BITE:", all_bit)
sys.exit(0 if all_bit else 1)
