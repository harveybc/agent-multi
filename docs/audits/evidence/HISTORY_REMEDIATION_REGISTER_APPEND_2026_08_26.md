# Append to the repository-history remediation item (finding 323 class)

Date: 2026-08-26. Authority: AUDIT_SATOSHI_CUDA_C0_V2_ACCEPTANCE
(DATA-SOTA-340 §4): "Register the already-published historical
exposure under the existing repository-history remediation item; tip
cleanup is not history erasure."

## Newly registered exposures (already published in git history)

1. **CUDA C0 v1 packet** (pushed at 39f3bc32, now tombstoned at tip):
   carried an 8-hex GPU-UUID fragment and an unsalted UUID sha256.
   Public body removed from the tip; restricted copy digest
   6c0ac31d...f70be1.
2. **CUDA C0 v2 packet** (pushed at 2ad94601, now tombstoned at tip):
   carried operator/interpreter/scratch absolute paths. Restricted
   copy digest b44ca301...36ba1.
3. **Legacy evidence files on the current tip**, inherited from
   pre-front eras and owned by OTHER fronts' audit records — listed in
   the scan's REGISTERED_LEGACY allowlist rather than mutated
   unilaterally:
   - ETH_EASY_ACTIVITY_SMOKE_2026_08_05.json (.local/state path)
   - HISTORICAL_FITNESS_PROVENANCE_GYMFX_8088F9E.json (host name)
   - MULTIFRONT_F1_L1_SAMPLE_2026_08_10.json (.local paths)
   - MUSASHI_LIVE_MODEL_IDENTITY_AFTER_241_2026_08_12.json (host
     names + /home path; Musashi-authored evidence)
   - MUSASHI_POST_OUTAGE_RUNTIME_FACTS_2026_08_11.json (legacy pre-front evidence with topology/UUID material)
   - P1LR_DECISION_FINAL_EVIDENCE_c0e53cf18b7d60dd_2026_08_15.json (legacy pre-front evidence with topology/UUID material)
   - PLATEAU_LR_CPU_SMOKE_2026_08_21.json (legacy pre-front evidence with topology/UUID material)
   - PLATEAU_LR_CUDA_SMOKE_2026_08_21.json (legacy pre-front evidence with topology/UUID material)
   - README_LINK_RESOLUTION_CHECK_2026_08_10.json (legacy pre-front evidence with topology/UUID material)
   - README_LINK_RESOLUTION_CHECK_2026_08_11.json (legacy pre-front evidence with topology/UUID material)
   - README_LINK_RESOLUTION_CHECK_POST_MERGE_2026_08_12.json (legacy pre-front evidence with topology/UUID material)
   - REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json (legacy pre-front evidence with topology/UUID material)
   - SOCIAL_ENRICHMENT_RETRY_DRYRUN_2026_08_10.json (legacy pre-front evidence with topology/UUID material)
   - SWARM_EFFICIENCY_MEASUREMENT_2026_07_31.json (legacy pre-front evidence with topology/UUID material)
   - SWARM_EFFICIENCY_MEASUREMENT_CLOCKED_2026_07_31.json (legacy pre-front evidence with topology/UUID material)
   - TOOLING_CYCLE_PROVENANCE_2026_08_06.json (legacy pre-front evidence with topology/UUID material)
   - WP2_ACTIVITY_PLATEAU_SENSITIVITY_DATASET_2026_08_20.json (legacy pre-front evidence with topology/UUID material)
   - WP4_CPU_SMOKE_REPORT_2026_08_20.json (legacy pre-front evidence with topology/UUID material)
   - WP4_REWARD_SCALE_CALIBRATION_2026_08_18.json (legacy pre-front evidence with topology/UUID material)
   - frag_dragon.json (legacy pre-front evidence with topology/UUID material)
   - frag_gamma.json (legacy pre-front evidence with topology/UUID material)
   - frag_omega.json (legacy pre-front evidence with topology/UUID material)
   Disposition of these files belongs with the finding-323 remediation
   decision (Musashi/owner), alongside the already-inventoried 13
   master-tip files.

The full finding-323 inventory lives at
`docs/audits/evidence/FINDING_323_UUID_EXPOSURE_INVENTORY_2026_08_25.md`
on branch `satoshi/post-p1-screen-b-20260825`.
