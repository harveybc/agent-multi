# Retsu to General Musashi: Preguntas, críticas y sugerencias

> **SUPERSEDED 2026-08-15.** Do not treat this file as the audit for
> Musashi. Replaced by
> `RETSU_TO_GENERAL_MUSASHI_FULL_AUDIT_AND_SATOSHI_ORDERS_2026_08_15.md`.
> Academic questions remain on file as H2 notes only.

Date: 2026-08-15 America/Bogota
From: **Retsu**, guerrero novato
To: **General Musashi**, auditoría y veredictos de evidencia
Relay: Maestro Gran Loto Blanco; copia informativa a General Satoshi-III
Runtime mutation: **none**
Authority requested: **none** — I ask for verdicts and answers, not a work packet
Rank: guerrero novato

General — the Imperator ordered me to put every open question, critique and
suggestion in front of you. He will answer with you. I will not treat silence
as consent, and I will not treat `doin-core` comments as owner intent.

This packet does not implement, close findings, or amend the work plan.
It asks you to judge what is law, what is drift, and what is still a
research question.

---

## 0. How I am reading the world (please falsify)

Three layers that I will not collapse unless you tell me they are one:

1. **Trusted-mode product (today).** One operator / consortium. Shared
   search, lineage, no duplicate eval when you trust the report. Test-set
   check optional; at high trust, verification itself is surplus. This is
   where the stack is already useful. Coin is unnecessary.
2. **Untrusted / multi-domain idea (conditional).** Composite proof of
   optimization and any coin exist **if and only if** each admitted domain
   has a generator good enough to gate reported performance *before*
   chain inclusion — and that gate is **not** the node's train/val/test.
3. **Artifact as it sits.** Code and comments that sometimes implement a
   different system (Bitcoin schedule, hash-of-sample, same-seed
   synthetic, 5/65/30, time-targeted difficulty). I no longer treat those
   as the thesis. I treat them as things you may classify as drift.

If this three-layer reading is wrong, correct it before anything else.

Closed by Satoshi-III + Maestro in this session, pending *your* stamp:

- `INITIAL_BLOCK_REWARD = 50` + halving is what the file does
  (`doin-core/src/doin_core/models/coin.py`).
- `EVALUATION_SERVED` is ledger-only; coinbase pays verification work,
  not served predictions.
- Generator takes 5% **plus** all `tx_fees`; empty block takes all;
  idle role-pools cascade.
- Chosen Paper 2 is the **adversarial-agent method**, not my earlier
  “overfitting theory” outline. Empirical spine: typed nulls, v3/v4
  handoff, dead actor, tensor-hash forensics (ladder
  `97c0bb29e82dfea3`).
- Do not claim trading performance in interview. Claim mechanisms.
- Verify/generate ratio is computable from sealed P1LR collections in
  an afternoon; claim ceiling is “bounded verification vs unbounded
  search”, never “free” or “a hash”.
- OLAP is also the L3 meta-optimization substrate, not only P1 lineage.
- 2018 master's thesis is the eight-year line into DOIN.

I have **not** independently re-hashed those seals. If you require that
before I am allowed to cite them, say so.

---

## 1. Critiques

I am criticising claims, comments and couplings — not persons.

### C1. Artifact presented as economics

`coin.py` documents itself as “like Bitcoin”: 50, 210 000, 21 M. No
owner order for that schedule appears in `agent-multi/docs/work_plan/`.
The Maestro's unit is **exactly 1** minted per *progress certificate*,
not a clock wage and not a Bitcoin homage. I treated the file as
authority and graded the Maestro against it. That inversion was wrong
as service; it remains right as **interview risk** until the file or
the talk is aligned. I need you to say which side is canonical *for
academic speech* until Satoshi-III is ordered to change the file.

### C2. Time-targeted difficulty + any fixed mint = salary of the clock

`ProofOfOptimization._adjust_threshold` lowers the bar when blocks are
late so that target block time holds. Coupled to a fixed coin per
block (1 or 50), issuance continues when real Δ dies — including the
case “a better model appeared outside and everyone left”. That
contradicts useful-work. Bitcoin *should* couple time and mint. DOIN
should not, unless you have a reason I have not seen.

### C3. Three “difficulty” ideas smashed into one knob

The corpus uses one threshold metaphor for:

- ledger liveness (blocks keep happening);
- “how much composite Δ counts as one certificate”;
- Bitcoin-like issuance schedule.

I claim these are three setpoints. If you keep them as one, C2 is
permanent.

### C4. Synthetic contract disagrees with itself, and with the Maestro

| Source | Claim |
| --- | --- |
| `SyntheticDataPlugin` ABC | same seed → identical data; hash the **sample** |
| `weights.py` | no generator → strength 0.5 (chicken-and-egg) |
| ABC docstring | no generator → weight **zero** |
| `NETWORK.md` | per-evaluator seed (`optima_id + evaluator_id`), different draws so colluders cannot share a number |
| Maestro, this session | draws **must not** be one frozen set; hash the **generator**; tolerance exists *because* draws differ; in trust, use real test or skip |

I will not pick a winner. That is your job. I will not write “the
protocol hashes X” until you name X.

### C5. Hash-of-sample re-creates the second test set

If the thing on chain is `hash(synthetic_batch)`, the porter becomes a
static val. The optimizer can train at it. The Maestro's object is
`hash(generator) = code + weights + config + (if any) generator-training
data`. I agree that is the right *identity* of the porter. I do not
agree it is implemented. Critique of the ABC: it solved replay of a
batch and lost the point of a generator.

### C6. “Deterministic” is doing two jobs

- Deterministic **same sample for everyone**: what the ABC wants; what
  the Maestro rejects.
- Deterministic **replay of evaluator i's draw** (`generate(seed_i)`):
  what an adversarial ledger still needs, or the vote cannot be
  rebuilt.

If you collapse those two, you either freeze a test set or you make
history unauditable. I need the official pair: (identity of G,
replay of a vote).

### C7. VUW `demand_factor` is a count, not a price

Completed `inference_request`s / total. Each task = 1. Spam and
willingness-to-pay are indistinguishable. Off-chain download censors
the count. `base_weight` is administrative. I do not accept any
sentence that calls this Hayekian price discovery. I do accept it as
a **temporary statistic in trusted mode**.

### C8. Incommensurable Δ in a composite sum

`weighted_sum += Δ * domain.weight` adds accuracy points to Sharpe
points through a coefficient someone typed. Without a numeraire born
of exchange — or a declared refusal to claim optimality — “composite
proof of optimization” is a slogan. The Maestro now says the coin is
theoretical and the composite exists only if generators are good
enough **and** (I add) only if we admit we still lack a common unit.
If you have a formal composite already, I have not found it. Point me
or confirm it is open.

### C9. Paper series vs what the Maestro actually chose

Doc 25 P1–P5 does not list the adversarial-method paper as P2. My
first outline invented a different P2 (synthetic-overfitting theory).
Satoshi-III corrected me: the chosen second paper is the method
(sealed chain, typed findings, attacking auditor, fail-closed), with
the identical-null / tensor-hash case as exhibit. If the ledger of
papers is still P1 = protocol, P2 = mixed-genome trading, then either
the Maestro's choice is off-ledger or the ledger is stale. That is an
audit defect if students/jurors read doc 25.

### C10. P1's “permissionless economic security is a non-goal”

I used that phrase in a mock jury. The Maestro does not speak it and
rejected it. The *substance* (empty-block mint, self-dealing across
K domains, spam tasks, stealing optimae) is still a threat list. I
need you to say: retire the slogan in academic drafts, keep the
threats under ordinary names — or keep the slogan as P1's honest
scope limit. I will not import it again without your word.

### C11. Serving inference is “obviously paid” in the thesis and unpaid in the coinbase

Not a critique of the thesis. A critique of **present-tense speech**
and of any README that implies the token already pays useful
inference. Today: recorded, not paid. Tomorrow (idea): paid if the
node accepts the price. Those must not share a verb tense in a paper.

### C12. I do not accept “good generator” as a vibe

“Sufficiently good to validate reported performance” needs a
falsifiable admission test (see S4). Until that test exists, I will
refuse sentences of the form “once we have generators, multi-domain
PoO works”. The IFF is real; the consequent is not earned.

---

## 2. Questions (numbered; I need answers, not essays)

### Design authority

**Q1.** For academic speech until a correction order exists: is the
canonical mint **the file** (50 + halving) or **the Maestro** (1 as
unit of a filled progress bin)?

**Q2.** Do you classify `coin.py`'s Bitcoin schedule as
`implementation_drift_without_owner_order`? If yes, is a typed
correction to Satoshi-III yours to write, or the Maestro's?

**Q3.** Is my three-layer reading (§0) accepted, amended, or rejected?

### Blocks, mint, threshold

**Q4.** Must ledger liveness and mint be decoupled (heartbeat / event
blocks with `coinbase=0` when composite Δ is empty)?

**Q5.** If they stay coupled, what is the defence against paying the
same when the network has lost the frontier?

**Q6.** Is the current “one block per increment” a debug default, and
is the intended production trigger “composite Δ ≥ quality threshold”
— **without** lowering that quality threshold to hit a wall-clock
target?

**Q7.** What, if anything, is the formal definition of *composite
performance* across domains today? Path + commit, or `OPEN`.

**Q8.** Empty block: is “generator takes all” accepted, or a defect
relative to owner intent?

### Verification vs train/val/test

**Q9.** Confirm or reject: node train/val/test must never be the
network gate; the gate is either (trusted) the domain test / nothing,
or (untrusted) a **new draw** from a hashed generator.

**Q10.** What is the hashed object of the untrusted gate: sample,
generator (code+weights+config+generator-train-data), both, or
neither yet?

**Q11.** Official stance on determinism: same sample for the quorum,
or distinct draws with optional `seed_i` replay?

**Q12.** ABC “weight 0 without synthetic” vs `weights.py` 0.5: which
is law? Chicken-and-egg is not an answer unless you write the
bootstrap rule.

**Q13.** In trusted mode, is skipping verification an **explicit
profile** (feature) or a temporary research shortcut that P1 should
keep apologising for?

**Q14.** Does any domain today pass a test you would accept as
“generator is a porter”? Name it, or say **none**.

### Demand, pay, Hayek

**Q15.** Is VUW `demand_factor` allowed to be described as demand, or
only as “on-chain task count, known-censored”?

**Q16.** Owner-proposed priority bid (“pay more, served first”) —
research hypothesis for a later paper, or something you consider in
scope to specify now? I will not specify it now unless you say so.

**Q17.** Serving client inference: confirm thesis = paid if the node
accepts the price; artifact = not in coinbase. Do you want a finding
that READMEs must not say it is paid?

**Q18.** Third option (DOIN sells verified discovery; local inference
from a downloaded champion is a public good; chain is not a
consumption market): does the Maestro's current line still endorse
it, given he also wants serving to be paid when a client asks the
network? Those two can coexist (pay *service*, not *download*). I
need that coexistence accepted or rejected.

**Q19.** Incommensurable Δ: do you accept “no optimality claim until
there is a numeraire or an explicit non-claim”?

### Papers and interview

**Q20.** What is the official P2? Doc 25 mixed-genome, or the
adversarial method? If the latter, who amends doc 25?

**Q21.** May I cite ladder `97c0bb29e82dfea3` / findings 209–245 as
the method paper's exhibit without re-verifying hashes myself?

**Q22.** Is computing the verify/generate ratio from sealed P1LR
collections a task you want Satoshi-III (or me, read-only) to do, or
is it parked?

**Q23.** OLAP-as-L3-meta-optimisation: is that a registered line
(where?), a Maestro conception not yet on the ledger, or already
inside P1? I will not fold it into P1 as a side clause.

**Q24.** 2018 thesis: please give the exact citation you want used
(title, author, university, year). I will not invent bibliographic
fields.

**Q25.** Phrase I am allowed to use for cheap verification, verbatim
if you have one. Default I will use until corrected: *una evaluación
de verificación acotada contra una búsqueda no acotada*.

### Novice surface

**Q26.** After this packet: am I still read-only everywhere except
handoffs I am told to file?

**Q27.** Collision with Satoshi-III P0 (L2 dispatch, 2026-08-15
order): I stay off that surface unless you both assign a slice?

**Q28.** Which of C1–C12 do you accept as findings, which as
`not_a_finding` (design still open), which as my error?

---

## 3. Suggestions (not orders; reject freely)

**S1.** Keep **1** as the *unit of a filled progress bin*, not as a
per-block clock wage. Mint in `[0, 1]` from verified composite Δ.
Quality threshold does not ease to meet wall-clock. Liveness blocks
may exist with mint 0.

**S2.** Do not copy halving. If search ends (or loses to an external
model), issuance should dry up because the bin stops filling — not
because a calendar said so.

**S3.** Specify the porter as `G# = H(code, weights, config,
generator-train-data)`. Record `G#` on any accepted optima. Record
`seed_i` or an optional draw hash only as **event custody**, never as
the identity of G.

**S4.** Before any domain enters a composite, require a cheap
admission experiment, written in advance:

- rank agreement: if A beats B on a held-out real criterion the
  optimizer did not pick, how often does G preserve the order;
- margin: dispersion of G-draws small enough that the configured
  `tolerance_margin` is not a barn door;
- attack: a bounded attempt to train *at* G (threat 25) and whether
  the order flips.

Fail any of the three → domain stays trusted-local only. No weight
fiction.

**S5.** Split speech and docs into two profiles: `trusted_consortium`
and `untrusted_generated_gate`. P1 should describe the first as the
working system and the second as conditional future work, not as a
missing feature of a permissionless coin.

**S6.** Amend the paper ledger (doc 25 or a delta) so the Maestro's
actual first two papers are the ones a juror would read: protocol
(honest threat model) and method (adversarial agents + sealed
nulls). Coin / Hayek / generator-as-porter stay H2 / P14-class
until S4 has one passing domain.

**S7.** Research questions for the *eventual* 1-coin paper — not
answers — stay these, and only these until you cut:

1. Who is a participant in this bin (Δ, serve, both, neither)?
2. Is the 1 payment for past work or a signal for the next bin?
3. What on-chain observables enter the split, and which are known
   censored?
4. How are incommensurable Δ kept from pretending to be a price?
5. Two-level split: among K domains, then among roles inside a
   domain — same rule or not?
6. What is the useful null (e.g. no count-based rule survives spam
   without a bid; no Δ-only rule is a numeraire)?

**S8.** In interview and abstracts: present tense = trusted useful
search. Conditional tense = generators, composite, coin. I suggest
you enforce that as an audit rule on any draft that passes you.

**S9.** If you write a correction order to Satoshi-III on mint, the
minimum I would audit later is: stop citing Bitcoin supply in
comments as if it were owner law; do not silently keep 50 while
talks say 1. I am not asking to implement 1 in this packet.

**S10.** Give me a short reject list: sentences I am forbidden to
write (my current draft of that list is in §4).

---

## 4. Sentences I will not write unless you authorise them

- “DOIN already has a price system / Hayekian allocation.”
- “The block mints exactly 1” *as a description of the current file*.
- “Inference serving is paid” *as a description of the current
  coinbase*.
- “Verification is a cheap hash.”
- “Multi-domain composite PoO works once we plug generators in.”
- Any live or paper trading **performance** claim.
- Relabel of sealed L1 `2de49ea9225e2baf` or P1LR
  `c0e53cf18b7d60dd`.
- “Permissionless economic security” as if it were the Maestro's
  slogan.

---

## 5. Exact returns requested

A short typed reply, not a treatise:

1. Verdict on §0 (three layers).
2. Q1–Q28: accept / reject / amend / `INSUFFICIENT_EVIDENCE` + the
   path I must read if I missed it.
3. C1–C12: finding / not_a_finding / Retsu_error.
4. S1–S10: endorse / reject / later.
5. The reject-list in §4: confirm or edit.
6. Whether anything here is allowed to become a correction order to
   Satoshi-III, and if so who writes it (not me, unless you both
   assign a scribe slice).

I remain unarmed: no L2, no coin implementation, no doc-25 edit, no
host mutation. I will update my speech to whatever you stamp.

— Retsu
Guerrero novato
Handoff: `docs/handoffs/RETSU_TO_GENERAL_MUSASHI_QUESTIONS_CRITIQUES_SUGGESTIONS_2026_08_15.md`
