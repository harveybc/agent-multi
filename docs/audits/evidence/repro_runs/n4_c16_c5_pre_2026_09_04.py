"""PRE freeze for order @9fd016b0 (C16, N4-C1..C5): executable
reproducers for every finding, before any edit."""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
s1 = importlib.util.spec_from_file_location(
    "n3f", REPO / "tools" / "n3_fresh_confirmation.py")
n3f = importlib.util.module_from_spec(s1)
s1.loader.exec_module(n3f)
s2 = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(s2)
s2.loader.exec_module(n4a)

print("== C16: stale authority docstring in verify() ==")
doc = inspect.getdoc(n3f.verify)
print("docstring mentions gate-bearing/allowlist:",
      "gate-bearing" in doc and "allowlist" in doc)
print("  quoted:", [line for line in doc.splitlines()
                    if "gate" in line or "allowlist" in line][:3])
assert "gate-bearing" in doc

print("\n== N4-C1: ternary license ignores class 2 ==")
src = (REPO / "tools/n4_target_audit.py").read_text()
assert 'support_classes = [0, 1] if kind == "class3"' in src
r = json.loads((REPO / "docs/audits/evidence/"
                "N4_SCREEN_RESULT_2026_09_04.json").read_text())
for ck in ("tm_h6", "tm_h12", "tm_h24"):
    sup2 = {wk: rec["class_support_score"].get("2")
            for wk, rec in r["per_window_records"][ck].items()}
    lic = r["per_candidate"][ck]["licensed"]
    print(f"{ck}: licensed={lic} class-2 supports={sup2}")
    assert lic is True
    assert all(v < 30 for v in sup2.values())
all2 = [rec["class_support_score"]["2"]
        for ck in ("tm_h6", "tm_h12", "tm_h24")
        for rec in r["per_window_records"][ck].values()]
print("true class-2 range across ternaries:",
      f"{min(all2)}-{max(all2)}  (the return packet WRONGLY said "
      "2-4 — disclosed count-discipline violation)")
assert (min(all2), max(all2)) == (2, 9)

print("\n== N4-C2: Holm family is 10, not the declared 14 ==")
holm_slots = sum(len(e["windows"])
                 for e in r["per_candidate"].values()
                 if e["licensed"])
print("p-values actually corrected:", holm_slots, "of 14 declared")
assert holm_slots == 10

print("\n== N4-C3: names overclaim ==")
print("family name 'tradeable_move' in tool/design/result:",
      "tradeable_move" in src)
print("arm 'target_history' fits barscale (volatility lags):",
      'hist_x = data["barscale"]' in src)
# semantics reproducer: an intrahorizon +5% touch that closes flat
closes = np.array([100.0] + [100.0] * 5 + [100.05])
plane = {"anchors": np.array([0]), "closes": closes,
         "highs": np.array([100., 105., 100, 100, 100, 100,
                            100.05]),
         "lows": closes * 0.999}
t = n4a.build_targets(plane)
print("intrahorizon +5%% touch, terminal +0.05%%: tm_h6 class =",
      int(t["tm_h6"][0]),
      "(class 2 no-trade — terminal-return semantics, NOT 'any "
      "trade within h')")
assert int(t["tm_h6"][0]) == 2

print("\n== N4-C4: design is named, not executed ==")
print("validate_design exists:", "def validate_design" in src)
assert "def validate_design" not in src
print("decision constants come from module constants "
      "(MARGIN/BOOT_SEED/...) — no design-vs-execution equality "
      "check exists")

print("\n== N4-C5: no standalone adjudicator ==")
print("adjudicate() exists:", "def adjudicate" in src)
assert "def adjudicate" not in src
print("licensing/p/Holm/verdict embedded in screen() only")

print("\nPRE CONFIRMED: all five findings reproduce")
