# T1 v4 independent reviewer invocation (logical roots)

From the agent-multi checkout at the return tip, with the accepted
T0 checkout available:

```bash
export B4_T1_PREPROCESSOR_ROOT=<preprocessor_checkout>   # at e6c3cdc
STATE=<state_root>                # operator-local evidence custody
DSHA=$(sha256sum docs/audits/evidence/T1_LAB_DESIGN_V4_2026_09_06.json | cut -d' ' -f1)
CUDA_VISIBLE_DEVICES="" python tools/t1_independent_verifier.py \
  --design docs/audits/evidence/T1_LAB_DESIGN_V4_2026_09_06.json \
  --design-sha "$DSHA" \
  --bank-dir  "$STATE/t1_bank_v3_20260906" \
  --npz-dir   "$STATE/t1_npz_v4_20260906" \
  --measurements "$STATE/t1_measurements_v4_20260906.json" \
  --measurement-manifest "$STATE/t1_measurements_v4_20260906_MANIFEST.json" \
  --published "$STATE/t1_adjudication_v4_20260906.json"
```

Expected: exit **3** with `SELF_CONSISTENT_ONLY_NOT_AUTHORIZING`,
`records_rederived_from_arrays: 1116`,
`complete_publication_equality: true`, and the digests in
`T1_V4_STATE_DIGESTS.json`. Exit 3 is the STRONGEST possible
candidate outcome by construction; the reviewed identity is your
separate record naming `measurements_sha256` and
`publication_sha256`.
