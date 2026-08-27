# Dispatch: One CPU Transfer-Loader Smoke

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Authorization: automatic consequence of accepted DATA-SOTA-353..356

## Build

Implement `tools/load_pretrained_branches_smoke.py` and the smallest reusable
loader library consistent with the plugin architecture. Do not place loading
logic in an experiment-specific driver.

The loader must:

1. Verify generation seal, complete v4 contract identity, source-data digest,
   ordered 83-feature partition, family digests, topology digest, code identity,
   preprocessing identity, origin-plan digest and purged-partition identity.
2. Resolve branches through installed entry points and reconstruct the declared
   grouped extractor without topology inference from checkpoint shapes.
3. Load **encoder state only**, by named family and exact key/shape/dtype.
4. Prove objective heads/adapters, optimizer state and calibration state cannot
   enter the transferred extractor.
5. Refuse missing, extra, renamed, reordered, duplicated or cross-family keys;
   refuse a valid tensor under the wrong family digest.
6. Re-serialize loaded encoder state and demonstrate bit-for-bit tensor parity
   with the sealed source state.
7. Run one finite CPU forward pass over an observation produced by the real ETH
   H4 `GymFxEnv` executing preprocessor, binding the observation and output
   shapes and ordered family identity.

## Required Adversarial Fixtures

- torn or substituted generation;
- v3 artifact offered to the v4 loader;
- one feature or family reordered;
- one parameter missing, extra, wrong dtype or wrong shape;
- two same-width family states exchanged;
- objective-head key injected into encoder state;
- optimizer/replay/calibration state offered as transfer state;
- runtime preprocessor or data digest drift;
- output containing NaN or Inf;
- a clean successful load repeated twice with identical output.

## Execution Boundary

After focused tests pass, execute exactly one CPU smoke with
`CUDA_VISIBLE_DEVICES=''`. Persist logical paths only, sanitize host/operator
identity, and return runtime, peak host memory, tensor parity, every family
digest, loaded/rejected key counts and forward-shape evidence.

The resulting artifact remains `MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE`.
Do not launch GPU work, compare economic performance, activate collectors or
promote any model. Return the implementation, PRE/POST adversarial evidence,
focused/full tests and smoke packet for independent audit.
