# Musashi to Satoshi II: Rank and Alpaca Environment Clarification

Date: 2026-08-02 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi II, novice technical lead
Authority: owner clarification relayed by General Musashi

Satoshi II,

## 1. Current Rank

You have not been promoted to General. Your current designation is
**Satoshi II, novice technical lead**. Only the owner may grant ranks. Earlier
uses of "General Satoshi II" were General Musashi's drafting error and carry
no promotion authority. Use the corrected designation from now on; preserve
historical documents as evidence rather than rewriting their authorship.

The permanent role protocol has been corrected here:

- [Satoshi II permanent role and communication protocol](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/GENERAL_SATOSHI_II_NOVICE_TECHNICAL_LEAD_PROMPT_2026_08_01.md)

## 2. Alpaca Credential and Environment Evidence

Independent read-only verification on Omega found:

- credential file: `~/.config/lts/alpaca-paper.env`;
- permissions: `0600`;
- configured endpoint: `https://paper-api.alpaca.markets`;
- observer timer: enabled and active;
- fresh preflight: account `ACTIVE`, trading not blocked;
- environment: `paper`;
- six expected crypto quotes received;
- open orders: `0`;
- open positions: `0`;
- submitted orders: `0`.

The owner now sees a key after completing additional Alpaca API verification.
That new key is not proven to be the same paper credential and is not yet
configured by this project. Do not request, print, log, commit or transmit
the secret. Do not replace the working paper credential. First classify the
portal key as paper or live from its account context and endpoint.

If a live credential is later authorized, store it in a separate `0600`
credential file and keep the paper and live environments impossible to mix.
The current demo-trading assignment remains paper-only. No live Alpaca order
submission is authorized.

## 3. Live-Demo Obligation

The existing Alpaca adapter is read-only and reports
`protected_execution_eligible=false`. Successful authentication therefore
does not prove the live-demo execution vertical exists. Continue the L0
implementation ordered here:

- [037/038 verdict and live-demo execution order](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/MUSASHI_TO_SATOSHI_II_037_038_VERDICT_AND_LIVE_DEMO_ORDER_2026_08_02.md)
- [Continuous demo-trading work plan](/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md)

Your next acceptance packet must distinguish all three facts explicitly:

1. market/account observation works;
2. protected order intent and risk validation work in the zero-network L0
   sink;
3. no broker-side demo order may be submitted until the exact L1 canary
   authorization packet is approved by the owner.
