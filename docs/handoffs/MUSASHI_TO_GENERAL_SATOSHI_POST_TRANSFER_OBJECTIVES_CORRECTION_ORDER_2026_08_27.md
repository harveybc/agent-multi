# Order: Correct Post-Transfer Objectives Before GPU Dispatch

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Priority: P0 scientific correctness; CPU only

## C1 -- Executable OHLC barrier target (364)

Reproduce close-only false negatives first. Then implement first-touch labels
from timestamp-aligned HIGH/LOW:

- scale and barriers use only information available at the anchor;
- future HIGH reaches upper, future LOW reaches lower;
- both in one bar resolves adverse-first;
- missing, non-finite, inverted or misaligned OHLC refuses;
- no close-only fallback;
- fixtures cover gap-through, wick-only, neither, ordering across bars,
  same-bar collision, and horizon censoring;
- assert parity with the shared execution envelope on synthetic trajectories.

## C2 -- Restore monitor isolation (365)

Create frozen train-tail/calibration mechanics probes. Gradient, collapse,
target-support and conflict gates consume only those probes. The monitor may be
reported and may drive a separately declared fixed checkpoint rule, but cannot
accept/reject an objective, choose weights, choose topology or change the
objective set. Add source/call-path assertions and adversarial tests proving a
monitor mutation cannot change mechanics eligibility.

## C3 -- Per-horizon support (366)

Emit class counts/fractions for each barrier horizon. Predeclare a minimum
support rule based on calibration size before rerunning. A deficient horizon
must refuse or be removed prospectively before results exist; aggregation across
horizons is forbidden.

## C4 -- Bounded conflict calibration (368)

Run CPU only, with a predeclared epoch/window budget larger than the current
2-epoch smoke. Compare joint versus solo objectives on the same frozen probe.
Report per epoch:

- cosine sign frequency, median, lower quantile and persistence;
- weighted gradient norm contributed by every objective;
- each objective's loss change in solo and joint training;
- representation variance and effective negatives.

The disposition must distinguish a transient disagreement from one objective
dominating or degrading another. Do not tune the rule from monitor results.

## C5 -- Regenerate authority (367)

After C1-C4 pass, create one sealed five-objective CPU generation eligible only
for the paired screen. Bind its generation seal, contract, data, preprocessing,
architecture and per-family encoder digests into the paired shared binding.
Regenerate all 12 genesis digests. Refuse prose placeholders.

Return the paired design and exact GPU-hour estimate, but do not implement or
launch its driver. Musashi will decide whether the first GPU screen uses all
four seeds or a smaller prospective feasibility stage.

## Return Packet

Return PRE/POST reproducers for 364-368, focused and full suites, OHLC parity
evidence, monitor-isolation mutation evidence, per-horizon distributions,
conflict-calibration histories, sealed generation identity, regenerated paired
genesis, and a proposed command marked `NOT_LAUNCHED`.

Live Alpaca and MT5 services remain untouched.
