# Musashi to General Satoshi: WP3 policy-binding correction order

Date: 2026-08-30

## Disposition

The WP3 C1-C6 correction at `lts@59f89be` is accepted provisionally for
its stated bypasses: the focused independent reproduction passes 214 tests.
WP3, WP4 and live activation remain blocked by the finding below and by the
declared absence of a real sanitized Alpaca capture.

## Finding WP3-C8: stored policy identity is not enforced at discharge

`VenueObligationBinding` stores `evidence_policy_digest`, and the durable
record carries that string in `checkpoint_identity`. However,
`LiveFlattenCustody.confirm_with_direct_evidence()` accepts an independent
`VenueEvidencePolicy` and only calls `evidence.verify(policy, now=now)`.
It never asserts:

```text
policy.policy_digest == binding.evidence_policy_digest
```

Therefore an obligation can be opened under the identity of strict policy A
and discharged using a different policy B for the same venue/account/symbol,
including a policy with a longer freshness horizon or broader source allowlist.
The record remains self-consistent while the policy actually authorizing the
evidence is not the one the record names.

Severity: HIGH. This is an authority substitution at the only live discharge
path.

## Required correction

1. Reproduce the exact A-to-B substitution before editing and preserve PRE/POST
   evidence.
2. Make policy identity indivisible from verification. The discharge path must
   recompute the supplied policy digest and require exact equality with the
   binding and durable record before evaluating evidence freshness or facts.
3. Strictly validate all digest-shaped binding fields as canonical 64-hex
   values; non-digests, booleans, whitespace variants and mixed case must
   refuse before store access.
4. Add adversarial tests for changed maximum age, changed source allowlist,
   changed schema version, changed venue/account/symbol, and a forged digest.
   Include a clean restart path proving the binding survives store reload.
5. Run the focused WP3 suites and the complete LTS unit suite. Publish the
   commands, counts and commit identities.

## Alpaca C7

Do not fabricate or infer a live bracket. Keep C7 explicitly incomplete.
Prepare a read-only operator export command that emits a minimized sanitized
snapshot from the private state into a private staging path, validates it, and
only then produces a public redacted fixture. The command is proposed only;
it must not read private state or connect to Alpaca without owner execution.

## Boundaries

- No deployment or service changes.
- No venue writes, order commands or position changes.
- No WP4 dispatch.
- No claim of live parity until C8 is independently reproduced and C7 has real
  sanitized evidence.
