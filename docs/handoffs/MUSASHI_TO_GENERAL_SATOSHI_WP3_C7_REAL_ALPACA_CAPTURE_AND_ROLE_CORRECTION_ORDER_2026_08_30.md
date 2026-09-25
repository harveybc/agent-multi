# Musashi to General Satoshi: WP3 C7 real Alpaca capture and role correction

Date: 2026-08-30

## Owner-authorized capture result

The owner explicitly authorized a read-only sanitized Alpaca export. Musashi
used the audited GET-only Paper client and performed exactly four successful
requests: account, clock, positions and open orders. No venue write method was
available to the client used. The minimized raw payloads were validated by the
current WP3 parsers and stored only in a private `0700` staging directory with
`0600` files. No credential, account identifier, broker order identifier,
private path or raw digest is published here.

Sanitized categorical facts observed:

- one SPY short position;
- one open SPY buy order;
- status `new`;
- `order_class=bracket`;
- `type=limit`;
- `position_intent=buy_to_close`;
- `legs` is null/empty in the open-order response.

The durable local lifecycle independently records a real bracket parent with
three broker ids and two protection legs. It is supporting provenance, not a
substitute for the direct payload above.

## Finding WP3-C9: a live protective order is classified as an entry

The current Alpaca open-order parser requires only `order_class`, `type`, side,
quantity and legs. It treats every top-level `order_class=bracket` object as an
entry and assumes the protective orders are present only inside `legs`.

The real Paper response disproves that assumption. Once the parent has filled,
the endpoint can return a resting protective child as a top-level bracket order
with no nested legs. Alpaca directly declares its semantics with
`position_intent=buy_to_close`; the current parser ignores that field and
therefore classifies this protective take-profit as an entry.

Impact: WIND_DOWN could try to cancel the wrong population or report a false
pending entry. Severity: CRITICAL for live weekly-flat activation.

## Required correction

1. Preserve the PRE result showing that the sanitized recorded shape is parsed
   as `entry` under the current code.
2. Add `position_intent` to the strict Alpaca order contract. Derive role from
   the venue-declared intent and order type, never from side/quantity geometry:
   opening intents are entries; closing bracket limit orders are protective
   take-profit; closing bracket stop/stop-limit orders are protective stop;
   any unsupported or contradictory combination refuses.
3. Support both real shapes explicitly: an unfilled bracket parent with nested
   legs, and a filled-parent protective child returned top-level with null or
   empty legs. Do not turn null into an empty list before contract validation.
4. Add a sanitized recorded fixture containing only synthetic identifiers and
   synthetic economic values while preserving the observed categorical shape.
   Its NOTE must distinguish observed categorical facts from substituted
   identifiers, prices, quantities and timestamps.
5. Test long and short positions, stop and take-profit children, unfilled
   parents, duplicate ids, contradictory intent/type/class combinations,
   missing intent and unknown intents. Assert that WIND_DOWN cancels only true
   pending entries and preserves both protective children.
6. Re-run the direct-evidence, adapter, custody, complete LTS unit suites, and a
   no-write dry-run using the corrected recorded fixture.

## Disposition

- C8 is accepted.
- C7 has now produced real evidence and is no longer blocked on owner action,
  but it is not accepted until C9 is corrected and independently reproduced.
- WP3, WP4 and live activation remain blocked.
- No service changes, venue writes or position changes are authorized.
