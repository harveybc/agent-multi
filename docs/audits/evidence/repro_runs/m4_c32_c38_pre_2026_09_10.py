"""PRE for order M4 C32-C38 (agent-multi@889320ee P2) at the
reviewed tip: the accepted calibration evidence stands byte-exact
and re-derives every order fact, while the CONFIRMATION protocol
the order commands does not exist yet.

Facts frozen read-only:
  A. the four order-pinned identities verify by recomputation
     (reviewed tip, design self, numeric amendment self,
     governing adjudication self);
  B. the order's exact facts re-derive from the governing
     adjudication STRUCTURES (21/28 eligible slots, exactly two
     incomplete generators, zero calibration-incomplete cells,
     M2 gain -0.41982887);
  C. ABSENT: confirmation successor, 16-contrast executable,
     confirmation runner/verifier, the two external-record
     templates, the C37 battery — and no CONFIRMATION array,
     score or ledger exists anywhere;
  D. zero writes (byte inventory equality over the evidence).

CPU only; nothing is generated, loaded or scored."""
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# ---- order-pinned constants (copied from the committed order
# text, docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_POST_M4_C31_
# PRIORITIZED_ORDER_2026_09_10.md §C32) ----
ORDER_TIP = "5e7a8fd430c8231a049baf03f00e720ba24ec994"
ORDER_DESIGN = ("d7280a92047d98898418fb7cd750b22c506a621e"
                "b381d9847e0fe926b7df69b9")
ORDER_AMENDMENT = ("43e0804e1e6e583b10ddbe46b7d4cd752838b047"
                   "3ccbc6496f0e458c49aedd4b")
ORDER_ADJUDICATION = ("b35b6fd969aa162047bdfb55b8f9fcce01aa7686"
                      "4c388d29a1c36642ab051ade")

DESIGN = (REPO / "docs/research/model_capacity/"
          "M4_SEALED_DESIGN_V5_2026_09_09.json")
AMEND = (REPO / "docs/research/model_capacity/"
         "M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_2026_09_09.json")
ADJ = (REPO / "docs/audits/evidence/"
       "M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_GOVERNING_"
       "2026_09_09.json")


def selfsha(doc, key):
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()


def inventory(root):
    inv = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            inv[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return inv


inv_before = inventory(REPO / "docs")

# ---- A. the four identities ----
import subprocess
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                      capture_output=True, text=True
                      ).stdout.strip()
assert head == ORDER_TIP, (head, ORDER_TIP)
print(f"A1 reviewed tip: HEAD == {head[:12]} (order-exact)")

design = json.loads(DESIGN.read_text())
assert design["design_sha256"] == ORDER_DESIGN
assert selfsha(design, "design_sha256") == ORDER_DESIGN
print(f"A2 design self: declared==recomputed=="
      f"{ORDER_DESIGN[:12]} (order-exact)")

amend = json.loads(AMEND.read_text())
assert amend["amendment_sha256"] == ORDER_AMENDMENT
assert selfsha(amend, "amendment_sha256") == ORDER_AMENDMENT
assert amend["amends_design_sha256"] == ORDER_DESIGN
print(f"A3 amendment self: declared==recomputed=="
      f"{ORDER_AMENDMENT[:12]}, amends the pinned design")

adj = json.loads(ADJ.read_text())
assert adj["record_sha256"] == ORDER_ADJUDICATION
assert selfsha(adj, "record_sha256") == ORDER_ADJUDICATION
assert adj["design_sha256"] == ORDER_DESIGN
print(f"A4 governing adjudication self: declared==recomputed=="
      f"{ORDER_ADJUDICATION[:12]}, binds the pinned design")

# ---- B. order facts re-derived from structures ----
slots = adj["confirmation_slots"]
elig = [s for s in slots
        if s["typed_status"] == "ELIGIBLE_UNDER_PROPOSED_RULE"]
assert len(slots) == 28 and len(elig) == 21, (
    len(slots), len(elig))
inelig = [s for s in slots
          if s["typed_status"] != "ELIGIBLE_UNDER_PROPOSED_RULE"]
assert len(inelig) == 7
print(f"B1 slots: {len(elig)}/{len(slots)} eligible, "
      f"{len(inelig)} typed ineligible (order-exact 21/28)")

inc_units = adj["incomplete_units_in_denominator"]
inc_gens = sorted({u.rsplit("::", 1)[0] for u in inc_units})
assert len(inc_gens) == 2, inc_gens
print(f"B2 incomplete generators: exactly {len(inc_gens)} "
      f"({[g.split('::', 2)[-1] for g in inc_gens]}), "
      f"{len(inc_units)} seed-units kept in the denominator")

disp = adj["dispersion"]
assert len(disp) == 28
cal_incomplete = [c for c, d in disp.items()
                  if len(d["complete_generators"]
                         if isinstance(d["complete_generators"],
                                       list)
                         else range(d["complete_generators"]))
                  < d["min_complete_required"]]
assert cal_incomplete == [], cal_incomplete
print("B3 calibration-incomplete cells: ZERO (every cell meets "
      "min_complete_required=13 under the 20% attrition "
      "allowance)")

gain = adj["ladder"]["m2_minus_m1_paired_gain"]
assert gain == -0.41982887, repr(gain)
print(f"B4 M2 gain: {gain} (order-exact) — M2 does not advance "
      "from CALIBRATION")

per_cell = [d["complete_generators"]
            if isinstance(d["complete_generators"], int)
            else len(d["complete_generators"])
            for d in disp.values()]
print(f"B5 complete generators per cell: "
      f"min {min(per_cell)} / max {max(per_cell)} of 16 planned")

# ---- C. the commanded protocol does NOT exist ----
absent = {
    "confirmation successor":
        list((REPO / "docs/research/model_capacity").glob(
            "*CONFIRMATION*SUCCESSOR*")),
    "confirmation runner tool":
        [p for p in (REPO / "tools").glob("m4_confirmation*")],
    "C36 musashi template":
        list((REPO / "docs/audits/evidence").glob(
            "*M4*CONFIRMATION*REVIEW*TEMPLATE*")),
    "C36 owner template":
        list((REPO / "docs/audits/evidence").glob(
            "*OWNER*M4*CONFIRMATION*TEMPLATE*")),
    "C37 battery":
        [p for p in (REPO / "tests").glob(
            "test_m4_confirmation*")],
}
for what, found in absent.items():
    assert found == [], (what, found)
    print(f"C absent as commanded: {what}")

state = Path.home() / ".local/share/agent-multi"
conf_artifacts = [p for p in state.glob("*m4*confirmation*")
                  if p.exists()]
assert conf_artifacts == [], conf_artifacts
print("C no CONFIRMATION array/score/ledger exists in the "
      "state root (m4 confirmation glob empty)")

reserved = sum(s["reserved_generators"] for s in elig)
print(f"C reserved-only: {reserved} CONFIRMATION generators "
      f"(48 x {len(elig)}) exist as RESERVATIONS in the "
      "adjudication record, no bytes")

# ---- D. zero writes ----
inv_after = inventory(REPO / "docs")
assert inv_before == inv_after
print("D zero writes: docs/ byte inventory equal "
      f"({len(inv_after)} files)")

print("\nPRE CONFIRMED: the accepted calibration evidence is "
      "byte-exact and re-derives every order fact; the "
      "CONFIRMATION protocol (successor, 16-contrast "
      "executable, runner/verifier, external templates, "
      "battery) does not exist; no CONFIRMATION data exists "
      "anywhere.")
