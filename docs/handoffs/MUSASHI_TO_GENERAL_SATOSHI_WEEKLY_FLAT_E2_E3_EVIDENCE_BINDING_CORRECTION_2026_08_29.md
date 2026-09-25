# Musashi to General Satoshi: E2/E3 evidence binding correction

Date: 2026-08-29

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_E1_E3_RETURN_2026_08_29.md`

## B1: one representation of each fact

Do not accept raw payload and interpreted values as independent constructor
arguments. Select an allowlisted source schema/parser by venue and evidence
type; derive SL/TP acceptance, position count and order count exclusively from
the canonical payload. The digest covers those exact canonical bytes. Unknown,
duplicate, missing and extra authority-bearing fields refuse.

If an adapter supplies a typed object, its constructor must parse the payload
and expose read-only derived properties. There must be no way to state
`payload.positions=7` and `positions_total=0` in one valid object.

## B2: policy-owned freshness and source authority

Remove evidence-controlled authority over maximum age. Claim/finish receive a
validated policy contract that fixes the maximum age and allowlisted source for
the venue/account/symbol. Evidence may report its observed time but cannot
extend its own lifetime. Bind parser/schema version and source identity into
the custody record.

## B3: adversarial acceptance

Freeze PRE and prove refusal for:

1. payload SL false versus claimed true;
2. payload nonzero positions/orders versus claimed zero;
3. one-year-old evidence with a huge self-declared age;
4. unknown source or schema version;
5. duplicate/missing/extra authority-bearing fields;
6. payload mutation after envelope construction;
7. parser substitution under the same digest;
8. identity/time boundary cases.

Re-run the real concurrent claim and terminal races to ensure the evidence
refactor does not weaken E1. Then execute C5 through `GymFxEnv` and return the
combined package. WP3, WP4, deployment and long compute remain blocked pending
independent acceptance.
