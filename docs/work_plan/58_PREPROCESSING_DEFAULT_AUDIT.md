# Effective Preprocessing Defaults: Read-only Audit

Owner concern2026-10-21: rolling z-score256 and other inherited transformations
may hurt learning. Audit the actual pinned code/config and distinguish a causal
implementation from an empirically appropriate transform. No active contract,
model, training process, source data or DOIN code changed during this audit.

## Findings

1. Rolling256 is real and active, not a dashboard label. gym-fx
   `preprocessor_plugins/feature_window_preprocessor.py` introduced mode
   rolling_zscore, window256 and clip10 in commitad8bbc41 on2026-05-03.
   The selected-input adapter retained and explicitly required mode/window in
   agent-multi9c286e0e on2026-10-19. Git metadata uses the shared repository
   author; it does not identify which assistant originated the choice. Musashi
   owns carrying this into the current experiments. Do not blame another agent
   without evidence or claim256 was selected for this SAC by an executed ablation.
2. For decision origin i (latest completed row), all13 sampled rows use one
   affine map per feature, fitted on raw hourly rows[max(0,i+1-256):i+1].
   These are up to256 observed H1 rows, not256 downsampled two-hour points.
   The latest completed bar is included, nothing later. The model's13x2 shape
   spans24 observed intervals but its preprocessing depends on longer history.
   Earlier window results therefore are conditional on this preprocessing, not
   an optimum independent of it. The documentation's step is an exclusive
   dataframe cursor; "before step" must not be confused with excluding t.
3. Every selected feature is scaled: binary-exempt list is empty. Includes
   RSI, oscillator levels and hod_sin/hod_cos/dow_cos. The cyclic fields are
   already bounded; their new center/scale change with history. This does not
   prove lost predictive utility, but does not preserve their fixed native
   coordinates and needs a separate comparison. Both hourly sine/cosine exist;
   weekday currently has cosine only, as the subset plan already records.
4. Clipping to[-10,10] is inherited from plugin defaults and is not explicit
   in the task config, although pinned code makes it reproducible. The fallback
   std<1e-8 ->1 is also active code. These are not harmless bookkeeping fields:
   clipping can remove extreme magnitudes. Real data confirms some clipping.
5. NaN/Inf replacement exists in the generic plugin, but this active contract
   requires finite complete windows AND finite scaler histories before calling
   it. The audited selected TRAIN/VALIDATION matrices are entirely finite. Do
   not claim missing-value imputation was used because a fallback exists.
6. Reward is equity change / initial_cash *1000, separately from observation
   scaling. This changes the learning reward, not the reported USD equity or
   returns. No second feature normalization/VecNormalize was found in the
   pinned SAC wrapper/build or temporal extractor path inspected here. No
   wavelet, centered filter or smoothing operation exists in this observation
   construction. This is not a new certification of upstream feature producers.

## Executed Evidence

Read only the independently collected closed13 control. TRAIN24862 rows,
VALIDATION6221 rows; input/manifests matched their contract hashes. Reconstructed
all complete13x2 observation windows:24838 TRAIN origins,6197 VALIDATION origins.
These diagnostic origins are NOT the business-entry mask and are NOT new
trading evaluations. Clipping counts are overlapping observation cells, not
unique source bars, trades or scored decisions.

| Split | Range cells | Return cells | Elapsed-hours cells | Total |
| --- | --- | --- | --- | --- |
| TRAIN | 16 | 19 | 1036 | 1071 |
| VALIDATION | 1 | 1 | 245 | 247 |

No clipping in the other12 fields. Cyclic fields are not clipped but can leave
their native range: VALIDATION maximum absolute standardized hod_sin1.61685,
hod_cos1.79435,dow_cos2.28958. That alone does not prove model harm.

Twelve direct pinned-plugin witnesses (six origins per split) exactly matched
the stated formula, float32 conversion and clipping. Replacing later data
left every tested observation byte-identical. Entire selected input matrices
and reconstructed standardized arrays finite. CPU only, no fit, no TEST.
The private reproducible probe and detailed per-feature output live in the
operator's separate preprocessing-audit state, not inside any closed run.

## Corrective Research, Not an Unannounced Live Change

Rolling scaling is not inherently an error, and a causal probe does not show
that it improves generalization. Treat256 as an unvalidated current choice,
not a mandatory consequence of causality. Do not substitute another arbitrary
rolling size or claim this explains losses without a controlled experiment.

Proposed next preprocessing contrasts, before adopting a final window/input
recipe: frozen TRAIN-only z-score versus rolling256 (other settings fixed),
then separately fixed native cyclic coordinates, then clipping off versus10.
Materialize all defaults explicitly in any NEW config and keep fitted scaler
statistics with the checkpoint. Reuse the same window/seeds/business/budget,
publish TRAIN/VALIDATION and drift/clipping diagnostics, TEST closed. Joint
checks follow shortlisted isolated contrasts; per-field transformations can
follow separately. None of those new contrast configs or scalers is currently
implemented or queued, and no result from them is claimed.

The active local12/13/14 shared-seed queue is untouched. Its conclusions remain
conditional on rolling256. Record this before interpreting the window winner.
