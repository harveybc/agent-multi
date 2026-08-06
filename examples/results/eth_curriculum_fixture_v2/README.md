# eth_curriculum_fixture_v2 (corrected, AUD-F1-20260805-114)

Discipline: train/train_tail/validation only; 2025 DISCLOSED, not evaluated.
report_sha256 80baee82f1fe7c8a5ae4d7d12fbbcf41b0ff9a1c070a5875e6c2163512401617

OPEN OBSERVATION (stated, not closed): easy and easy_normal report
identical validation metrics to full precision in this rerun. Plausible
cause: under the corrected validation-only checkpoint rule, the
easy_normal normal phase did not improve on the post-easy checkpoint,
so the selected weights are the post-easy artifact itself; ETH data in
this budget raises no would-margin-call events, making easy and normal
training dynamics identical. Requires explicit confirmation before any
curriculum claim is made from this fixture.
