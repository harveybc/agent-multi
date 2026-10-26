# Native wrapper sensitivity audit

Owner requests ongoing critical audit and authorizes corrective measures. Do
not change the running experiment's code, data, weights, support or cost rules.
Diagnostic output lives outside that campaign and never enters its ranking.

## Tests and evidence

1. Authenticate completed requests, recipes, checkpoints and native artifacts;
   compare distinct initial/final policy tensor hashes, not ZIP timestamps.
2. Measure categorical probabilities, action diversity and policy/value head
   saturation on deterministic samples of common valid TRAIN and INNER_A
   windows. Flat/long/short masks are hypothetical probes, not a new simulation.
3. Compare zero-at-TRAIN-mean, repeated-current-tick, reversed-window and
   reordered-window probes. These are diagnostic perturbations only; input
   transforms in the scientific run remain unchanged.
4. Measure input gradients of command margins. A constant-command positive
   control must be identified even when the encoder varies; an unsaturated
   policy must respond. No optimizer updates; weight hashes must remain exact.
5. Replay simple mask-only long/HOLD and long/immediate-close rules through the
   corrected engine for the appropriate completed controls. Compare trades,
   daily equity and metrics exactly; this tests a behavioral explanation, not
   a new performance claim or selected strategy.

The flat/held probes are not trading states reconstructed from actual open
orders. Perturbation responses do not establish profitable information or
absence of information. No learned feature clipping, rolling normalization,
feedback inputs, expert targets, TEST, or changes to early stopping are added.

Decision: leave the queue untouched while collecting evidence. If measured
collapse prevents the wrapper from distinguishing candidate utility, preserve
completed receipts and pause at a documented boundary rather than consuming
the entire budget without a useful comparison. Any correction needs separate
identity and controlled verification before restarting financial selection.
