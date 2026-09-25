# Musashi audit: M4 C31A-C31F and external campaign status

Date: 2026-09-10

Reviewed tip: `5e7a8fd430c8231a049baf03f00e720ba24ec994`

Disposition:

`M4_CALIBRATION_ACCEPTED_WITH_PRE_CONFIRMATION_FREEZE_REQUIRED`

This accepts the corrected CALIBRATION evidence and its negative M2 gate. It
does not authorize CONFIRMATION. The B4 service exit is not accepted as a
completed campaign.

## 1. Inputs and identities

- sealed v5 design file SHA-256:
  `0a0fb757d847749373259031cf99f0844d13fb37a8d135462c9f926b1930700a`;
- design self-identity:
  `d7280a92047d98898418fb7cd750b22c506a621eb381d9847e0fe926b7df69b9`;
- numeric-validity amendment file SHA-256:
  `49c363953641ea7be2eff124654cf861d4214402ab921b7276d7a768ced61566`;
- amendment self-identity:
  `43e0804e1e6e583b10ddbe46b7d4cd752838b0473ccbc6496f0e458c49aedd4b`;
- governing adjudication file SHA-256:
  `51247b7853f7da1d5e549810f6b680183e7f4a4c77a4c787188030b6e560ff4c`;
- governing adjudication self-identity:
  `b35b6fd969aa162047bdfb55b8f9fcce01aa76864c388d29a1c36642ab051ade`;
- final return packet SHA-256:
  `49e23f0312fb1713ca4b9a11a7ef5f72ccbf1b3b746af332055237d1293a40c0`;
- incident POST output SHA-256:
  `1185512881137706c6d8b1b1b11d92dbca2121d68860876f21032a621fccce2c`.

The numeric amendment and corrected implementation were committed and pushed
at `7169552118859f2874a473f85f80c38bd7254fab` before governing attempt 3.
The branch was clean and its tip matched origin at audit time.

## 2. Independent checks

1. The focused batteries were rerun at the reviewed tip:
   `45 passed` across `test_m4_numeric_incident.py`,
   `test_m4_v5_protocol.py` and `test_m4_intervention.py`.
2. A fresh execution of `verify_run_v5()` against the physical attempt-3 root
   reconstructed `1,984` screen units and `1,344` intervention units and
   returned `verified: true`.
3. The adjudicator was rerun from the verified physical summaries. Its complete
   output equaled the committed governing adjudication exactly, including its
   self-identity.
4. The derived facts are therefore accepted as CALIBRATION facts:
   83/116 screen cells satisfy the proposed rule; 21/28 reserved
   family/noise/width slots satisfy it; all eligible slots have dispersion
   UCB95 at or below six; two generators are incomplete and remain in their
   denominators; no cell crosses the sealed attrition floor.
5. The prediction ladder is an accepted negative calibration gate:
   M0 `0.00326826`, M1 `0.00993885`, M2 `0.42976772`, paired M2-minus-M1 gain
   `-0.41982887`. M2 does not advance.

## 3. Findings

### F1 - The 12/16 rule is calibration-derived, not predeclared

`PROPOSED_ELIGIBILITY` is not present in the sealed v5 design. The adjudicator
that defines it was added after the original pre-outcome boundary and describes
the rule itself as not sealed pre-outcome. The CALIBRATION population is the
proper place to choose and freeze such a rule, and CONFIRMATION remains
untouched, so this does not invalidate attempt 3. It does prohibit calling the
rule predeclared.

Before CONFIRMATION, a successor must identify it truthfully as
`CALIBRATION_DERIVED_AND_REVIEWED`, bind the complete attempt-3 adjudication and
freeze the exact rule: at least 12 of 16 learnable outcomes, zero numerically
invalid outcomes, and the already-derived 21 eligible slots.

### F2 - The confirmatory estimand is underspecified across widths

The sealed multiplicity family contains 14 intervention contrasts indexed by
family and noise, while the executable reservation contains 28 slots because
each contrast has widths 16 and 64. The confirmatory protocol must state before
any CONFIRMATION array exists how the two widths contribute to each contrast,
including what happens when only one width is eligible. Width-specific outcomes
cannot silently become 28 hypotheses under a family of 16.

Recommended rule: within each generator, average the paired intervention effect
equally over the eligible widths frozen by CALIBRATION; one eligible width uses
that width; no eligible width produces a non-rejecting `NOT_EVALUABLE` slot.
Always report width-specific effects as secondary heterogeneity results.

### F3 - M2 must remain stopped

The M2 descriptor ladder failed by a large margin. No M2 model may be fit on
CONFIRMATION. Preserve the frozen `incremental_prediction::M2_vs_M1` contrast
as a non-rejecting placeholder (`p=1`) so removal does not shrink the
multiplicity family after seeing CALIBRATION. Descriptor measurements may be
recorded only if they are already required for the primary intervention and
their cost remains visible; they cannot regain selection rights.

### F4 - There is no confirmatory executor or authority gate yet

The reviewed runner explicitly refuses every role other than DEVELOPMENT and
CALIBRATION. That is correct. A separate successor, exact population census,
independent verifier, analysis implementation and external execution record
must exist before CONFIRMATION is generated or scored.

### F5 - B4 did not complete

`systemd` reports the manually stopped B4 service as `inactive`,
`Result=success`, exit status zero. That is a service-manager fact, not campaign
completion. The physical v7 root contains exactly two cell terminals with two
complete seals, three claims total, an ambiguous third claim with no terminal
or complete seal, and nine unclaimed cells. The owner stop markers remain.

The statement that B4 completed is therefore withdrawn. The prior B4 C49-C56
recovery order remains P0 and no B4 restart is authorized.

## 4. Accepted boundary

- Governing M4 attempt 3: accepted as corrected CALIBRATION evidence.
- Eligibility: accepted only as a calibration-derived candidate freeze.
- M2: accepted negative; does not advance.
- CONFIRMATION: closed pending a reviewed successor and external record.
- T2: execution completion remains awaiting C89-C94 reconstruction and
  adjudication.
- B4: quarantined incomplete campaign; C49-C56 remains mandatory.
