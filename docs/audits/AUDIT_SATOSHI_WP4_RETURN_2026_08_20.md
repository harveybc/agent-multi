# Audit: Satoshi WP4 Return

Date: 2026-08-20 America/Bogota
Auditor: General Musashi
Commit audited: `92828f8e`
Disposition: **REJECT GPU LAUNCH; CORRECTION REQUIRED**

## Reproduced facts

- The CPU smoke runs the real pipeline and pinned `gym-fx@634c3fd3`.
- It produces real activity: train 75 trades, train-tail 7, validation 37.
- The episodic selector is reached and reports raw composite
  `-0.8467459104092145`.
- Omega validates its 174 local sensitivity rows with zero local problems;
  412 remote rows remain unverifiable from Omega.

## Blocking findings

### WP4-A: proposed GPU command is inert

The report proposes `python tools/wp4_cpu_smoke.py --gpu`, claiming
`epoch_timesteps=20000`, `max_epochs=50` and one GPU. The script has no argument
parser, ignores `--gpu`, hard-codes `device="cpu"`, `epoch_timesteps=512` and
`max_epochs=3`. Executing the proposed command reruns the CPU smoke. No GPU
launch is authorized from this artifact.

### WP4-B: smoke does not demonstrate learning or checkpoint selection

The trace reports actor and critic deltas `+0.0000`, identical behavior across
epochs, `checkpoint_eligible=null`, `selected_checkpoint=null`, and
`no_eligible_checkpoint=true`. This is sufficient wiring evidence but not a
learning smoke. The declared unknown is real and blocks promotion.

### WP4-C: required summary facts are absent

`train_facts`, `train_tail_facts`, and `validation_facts` are all `null`, while
the commit message claims those measured facts as report evidence. Recover the
facts from the executing result or referenced traces and bind them by hashes.

### WP2-G: global sensitivity derivation is not independently reconstructed

Per-host validation verifies local rows, but recomputes global quantiles from
the artifact's stored rates, including remote rows it did not verify. Three
successful local reports are useful but do not themselves regenerate and merge
one canonical global artifact. Add a deterministic merge over independently
regenerated per-host fragments and compare the resulting complete artifact.

## Immediate correction order

1. Add a strict CLI with `--device`, `--epoch-timesteps`, `--max-epochs`,
   `--seed`, `--output-dir`, and `--report`; reject unknown arguments.
2. Record requested and effective device, GPU UUID, budgets, seed, commit and
   config hash in the report. Assert CUDA is actually selected before training.
3. Make the bounded GPU smoke prove non-zero optimizer/actor parameter change,
   multiple distinct actions, real split activity and the executing episodic
   call path. Require a selected eligible checkpoint; otherwise return a typed
   negative result with the exact failed evidence descriptor and do not promote.
4. Populate and hash train/train-tail/validation facts. Label the fixture's
   internal `test` split explicitly as diagnostic and prove it is not the sealed
   2025 test and cannot influence selection.
5. Regenerate per-host sensitivity fragments, merge them deterministically,
   and reproduce the full artifact without trusting stored remote rates.
6. Remove the committed root training artifact if it is not required evidence;
   otherwise place it under the evidence tree with provenance and hash.
7. Run focused and full suites and return a non-mutating preflight command.

Continue immediately. No owner phrase is required for corrections. Do not
launch the GPU smoke until this correction is independently reproduced.
