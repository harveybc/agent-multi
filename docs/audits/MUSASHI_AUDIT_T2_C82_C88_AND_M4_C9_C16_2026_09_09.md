# Musashi audit: T2 C82-C88 and M4 C9-C16

Date: 2026-09-09

## 1. Scope and identities

Reviewed independently:

- T2 tip `7bcd3f0d6a95b30ce105df3b1e7e13614b7fe6b3`, tree
  `6c2942a24ef4523447801a179cf5f15c9da5617b`;
- M4 tip `318cfdb8da0721d46f6da734c7c51659ad86dd4d`;
- T2 resource-successor file SHA-256
  `0e0317e2de947fd49efec10e4849909e25f619174533863520301c473a81a97f`;
- T2 resource-successor self identity
  `0542ed0536c38e92707454f0e55134246dddecbfbad4a63db1817c1ba1ce4e27`;
  and
- M4 mechanics design v3 self identity
  `cf4f429327933037531716c47c7dd6d036c020de0845689821dd427b2fcae013`.

Both reviewed worktrees were clean and equal to their pushed branch tips.
M3 evidence was not regenerated. B4 was observed read-only and was not used as
a source checkout.

## 2. T2 findings and reproduction

No blocking finding remains in C82-C88.

The corrected flow freezes manifest, census and active design from one
descriptor-bound stream and passes those snapshots through planning and
execution. The productive entry point no longer reparses or resolves the
active design after the gate. The resource-only v6-to-successor comparison is
also repeated immediately before durable effects. The successor requires its
own external execution record and cannot consume the old v6 record.

Independent results:

- committed POST: the original replacement adversary produced 242 units under
  design A, while the previous vulnerable behavior produced 3 units under B;
- a scientific delta in the successor refused;
- restoring the second read and removing the final comparison reopened the
  exact adversary, proving that both protections are load-bearing;
- focal T2 battery: **95 passed, 1 skipped**, 30 warnings, 779.82 seconds; and
- the worktree remained clean after reproduction.

Disposition:

`T2_C82_C88_ACCEPTED_AND_SUCCESSOR_EXECUTION_OPENED`

Musashi authored the real external successor record only after the independent
reproduction. Its SHA-256 is
`4761df4a8b324442a83a0554cc53d3005025500d9e8d0c3b2272a729c9714752`.
It pins the reviewed successor bytes, the existing design-review record,
manifest, census, complete executor code surface, commit and tree. The record
is local reviewer authority, mode 0600, and is not committed to Git.

The real zero-write plan then returned:

- 242 PENDING, 0 COMPLETED, 0 TERMINAL and 0 UNCERTAIN;
- 7,744 selected model instances;
- 23,232 MLP candidate fits;
- 1,936 linear solves; and
- 484 baseline evaluations.

The output root was absent both before and after `--plan`.

## 3. T2 launch

The bounded CPU campaign was launched as
`t2-confirmatory-resource-successor-20260909.service` from the exact reviewed
checkout. Execution has CUDA hidden, process nice 15, one logical worker, an
8-GiB RSS limit and the reviewed 216,000-second hard wall.

First post-launch observation:

- service `active/running`;
- `NRestarts=0`;
- four verified unit records complete and a fifth claim in flight;
- private results root mode 0700; and
- heartbeat and wall ledger advancing.

This is a public non-financial transformation screen. Its outcome can license
or reject the causal EWMA operator for later domain validation; it does not
alter B4 or grant financial/live authority.

## 4. M4 findings and reproduction

No blocking mechanical finding remains in C9-C16.

Independent results:

- committed POST: all six forged-evidence adversaries refused;
- all four targeted guard mutations re-admitted their specific adversary;
- focal M4 battery: **19 passed** in 11.46 seconds; and
- direct fresh verification returned `verified: true`, two units, 20
  artifacts, causal restart true for both units and matched-compute difference
  0.0.

The v3 correction is science-neutral and correctly matches 16 diagnostic
examples to the primary update. Its endpoint remains cumulative and its
verifier reconstructs transitions instead of trusting producer verdicts.

Disposition:

`M4_V3_RECONSTRUCTIBLE_MECHANICS_ACCEPTED`

The mechanics do not yet constitute a confirmatory intervention. The existing
design lacks the exact affordable population, explicit estimands for endpoint
prediction, a precision calculation, frozen per-family learnability thresholds
and a complete scientific runner. These are design duties, not defects in the
accepted two-unit mechanics.

## 5. Next order

Satoshi is ordered to execute M4 C17-C24 under:

`MUSASHI_TO_GENERAL_SATOSHI_M4_C17_C24_INTERVENTION_FOUNDATION_ORDER_2026_09_09.md`

That order opens exact population design, generator custody, the
DEVELOPMENT-only optimizer learnability gate, precision/multiplicity design,
runner implementation, adversarial tests and four bounded CPU mechanics units.
It preserves untouched CALIBRATION/CONFIRMATION outcomes and grants no GPU,
DOIN or production authority.

## 6. Runtime and owner boundary

B4 v7 remained active with zero service restarts. Read-only observation found
two completed cells and the third training, with the GPU fully occupied. A
thermal reading touched the declared 87-degree boundary; no process was
modified, and B4's own runtime guard remains authoritative.

No owner action is required in this cycle. T2 is running; M4 work is assigned;
M3 is closed; B4 continues under its existing service. The inherited D1 pair
remains unrelated operator cleanup and blocks none of these paths.
