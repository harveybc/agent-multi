# Musashi to General Satoshi: Post-P1 Baseline Execution Order

Date: 2026-08-25 America/Bogota  
Priority: Front 1, immediate  
Owner intent: advance useful compute; do not repeat inert P1

## 1. First reproduce the terminal result

Run the commands in
`AUDIT_P1_L1_CURRICULUM_TERMINAL_AGGREGATION_2026_08_25.md`. Verify all
12 report hashes, the path-to-digest normalization and all eight 148/148
state comparisons. Do not close P1-316/317 yourself.

## 2. Correct P1-316 and P1-317

1. Make `nested_split_contract_sha256` authoritative in future arm and
   pair contracts; absolute paths are descriptive and excluded from pair
   identity.
2. Bind each selected state map into the terminal arm report by digest and
   either embed it or copy it into the evidence packet. Aggregation must
   refuse a missing, malformed or hash-mismatched map.
3. Add regression fixtures for path relocation, changed contract content,
   absent state map, changed tensor and duplicate/missing arm.

## 3. Execute Screen B rule arms now (CPU)

Implement and run B0, B1, B2a, B2b and B3 exactly as doc 40 specifies,
through the same GymFx cost, SL/TP and action-accounting path. Use all three
causal origins (score 2022/2023/2024); 2025 must be structurally absent.

Required before execution:

- formula and lag tests for every arm, including no t information entering
  a position decided at t;
- identical scored indexes and cost-config digest across arms;
- B0 demonstrates zero exposure and zero costs without manufacturing a
  favorable economic score;
- B1 enters causally and reports initial turnover/cost;
- B2 lookbacks are exactly 180 and 540 H4 bars using `close[t-1]`;
- B3 uses target 15%, realized-volatility window 180, lag 1,
  annualization sqrt(2190), leverage cap 1;
- persist per-bar net returns, position, turnover, costs, envelope/policy
  close reasons and all hashes required by doc 41;
- pre-register every arm/origin row in the trial ledger before results.

Run CPU arms immediately after focused and full tests pass. Publish raw
results and the doc-41 descriptive block, but do not claim G1: B4 is not yet
present.

## 4. Materialize B4, do not launch it yet

Prepare the four-seed, three-origin causal SAC plan using fresh genesis per
origin and the approved direction `ethusdt_4h_l1_system_v2`: exact ordered
83-feature contract, `typical_price` excluded, price window false, flattened
dimension 2,660. P1's executed 84-feature artifacts are diagnostic only and
must never warm-start B4.

The materializer must prove before model construction:

- fit and selection end before each score origin;
- observation list/order/count/digest and flattened shape;
- equal cost/action/stopping recipe across seeds and origins;
- 2025 absent;
- fresh zero-update genesis artifact and hashes;
- expected GPU-hours from a bounded CPU smoke and one proposed GPU preflight.

Return B0-B3 evidence plus the B4 materialization for independent audit.
Do not dispatch B4 GPUs until Musashi verifies this packet. Other useful CPU
work continues in parallel; live Alpaca and MT5 services remain untouched.
