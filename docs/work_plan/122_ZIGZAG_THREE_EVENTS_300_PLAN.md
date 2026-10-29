# Three-event selection and controlled 300-generation refinement

Owner delegated K choice and requested 300 generations, 2026-10-28.
Select K=3: preserves the first additional-exit gain over K=2 while avoiding
much of the complete-origin coverage loss from K=4/5, especially at 24 h.
All four variants remain documented; this is an engineering selection under
the current observed-session support, not proof of a universal optimal K.

## Frozen experiment

- K=3; horizons 3/6/12/24 h; population 20; seeds 7/19/43.
- Increase generations only: 30 to 300. Do not add champion-seeded initial
  candidates, change costs, risk floors, support, horizon bands or operators.
- Fresh evaluation of all genomes in a new root, no historical cache reuse.
- Authenticate historical baseline receipts and snapshot all its file hashes.
- Current evaluator differs from the baseline revision by allowing K=4/5;
  the K=3 evaluator must reproduce all 30 historical generations exactly
  for every seed/horizon. A mismatch invalidates the paired comparison.
- Revalidate pinned inputs, engine and software, complete scalar audit,
  require elitism non-regression and verify baseline files unchanged.
- Report actual 300-generation champions, per-seed improvements, common-origin
  comparison, coverage, direction accounting and measured elapsed time.

## Automation

`tools/extend_zigzag_opportunity.py --fresh --baseline <completed-next3-root>
--out <new-three-event-300-root>` executes the entire search, scalar audit and
comparison serially. Population and seeds are taken from the authenticated
20x30 baseline. The new root seals current code plus this plan. No GPU,
financial-model training, validation/test read or floating selection occurs.

The 24h/K=5 known-control counterexample remains documented in 121. It does
not justify silently injecting controls into this otherwise paired K=3 run.
