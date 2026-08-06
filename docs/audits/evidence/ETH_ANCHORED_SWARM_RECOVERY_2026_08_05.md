# ETH Anchored Swarm Recovery Evidence

Date: 2026-08-05 America/Bogota

## Accepted Smoke

- Domain: `trading-asset-policy-eth-4h-anchored-smoke-v1`
- Seed: `2703`; population: `4`; distinct workers/candidates: `4/4`
- Canonical tip after deterministic fork choice:
  `13fbfbe5369f06d1d8562db1e62bec3f96ac894c66a6028c412f8b51e09d74d3`
- Champion: 17 normal-validation trades, total return `0.009695368029396079`,
  maximum drawdown `0.050749762294270015`, final equity
  `10096.95368029396`.
- Model artifact: 33,705,132 bytes, SHA-256
  `4da5de5eaa2e7455130ea36b1a7d14f65007d9ea3b4d5ad8d556a0879a6e4230`.
- Parameter artifact SHA-256:
  `807c2eb60f003ce232584e52ad45cf225d52bbfcb6ee0c3eb278eb7ce74b5788`.

## Reproduced Defects And Disposition

1. The champion-bearing block response was approximately 45 MB. Gamma needed
   about 30 seconds to transfer it from Dragon while `fetch_blocks` had a
   30-second total timeout. `doin-node@9eba394` assigns block transfer a
   dedicated 120-second bound; focused tests: 83 passed; full suite: 409
   passed. All four smoke nodes then converged to the canonical tip above.
2. Anchored `full-v1` rejected candidate indices 2 and 4 with
   `evaluation_error: 'policy.optimizer'`. Both selected automatic entropy
   while the source SAC archive used fixed entropy. The run was stopped after
   those two results and is not resumable scientific evidence.
3. `agent-multi@7c49e5f7` constructs the candidate SAC first and transfers the
   complete champion policy state without optimizer moments. Fixed-to-auto
   entropy initialization is preserved as `auto_<source coefficient>`.
   Focused suite: 23 passed; full suite: 539 passed; the dedicated loader suite
   also passed 5/5 on Dragon and 5/5 on Gamma.

## Active Full V2 Contract

- Plan: `phase-2-eth-anchored-full-fleet-v2`
- Canonical plan hash:
  `eede6f8650dbc85e497b27c89185b67d97a99a47dab56f8402ebe674437ce16d`
- Domain: `trading-asset-policy-eth-4h-anchored-full-v2`
- Semantic hash:
  `a422e828098708d6fe6ea97a3de6a5f6f1f7518eb3aaf193c632121b8548a077`
- Dataset SHA-256:
  `1b447c66e68495e826c53e2ab2b08ecd3922c8fdc735747628f8d0435ebe440f`
- Initial shared tip:
  `22e0f3141739f404796830fe77e3be170c98bcd05b1c65b2752fbe4c962cbc1d`
- Initial claims: Gamma-5070Ti index 0, Dragon index 1, Omega index 2,
  Gamma-5090 index 3; no duplicate claim and no rejected evaluation.
- Omega index 2 is the previously failing automatic-entropy chromosome. Its
  first completed epoch recorded 102 train trades and 61 validation trades;
  validation return was `0.024643` with final equity `10246.43`. This directly
  proves that the corrected loader crossed the former `policy.optimizer`
  failure point and that activity did not collapse to zero.
- Fleet revisions: `agent-multi@5437a31`, `doin-node@b70ea03`,
  `doin-core@e05a332`, `doin-plugins@8c959a6`, `gym-fx@9a084ac`,
  `trading-contracts@cd05083`.
- Planned work: 20 candidates per generation; 8 model-training, 4
  execution-risk and 6 joint-refinement generations (360 nominal candidate
  evaluations before early stopping).

The accepted smoke, rejected `full-v1` and active `full-v2` use separate state
and artifact roots. No chain database was deleted or rewritten.
