# Social and Multi-Venue Activation Instructions

Date: 2026-07-30
Owner actions only; never paste a key, password, account ID or recovery code
into chat, Git, shell arguments or tracked JSON.

## 1. Moltbook

The public collector, OLAP, low-cost triage and supervising Telegram review
are already active and the end-to-end Telegram delivery has passed. Choose
exactly one identity path to activate identity-aware participation.

### Reuse an existing agent

```bash
cd ~/Documents/GitHub/agent-multi
./examples/scripts/configure_moltbook.py
systemctl --user start agent-multi-social-collector.service
```

The script asks for the existing key without echoing it.

### Register a new agent

```bash
cd ~/Documents/GitHub/agent-multi
./examples/scripts/register_moltbook.py
```

Open the printed official `https://www.moltbook.com/claim/...` URL and complete
owner verification. Registration writes the key locally with mode `600`.

Verify:

```bash
python tools/social_intelligence.py \
  --config examples/config/social_intelligence/moltbook_observe_v1.json status
systemctl --user status agent-multi-social-collector.timer
```

Publishing is disabled. An approved publication follows:

1. create a source-backed draft;
2. approve that exact draft as the human owner;
3. enable publishing in a local reviewed config;
4. publish at most one draft;
5. when Moltbook returns a challenge, solve it within five minutes and run
   `verify-draft` with the answer formatted to exactly two decimals.

No social content can place a trade or change a DOIN campaign.

The system will continue collecting, triaging and reporting without a
Moltbook credential. Creating drafts or publishing requires one of the two
identity steps above; there is no valid unattended workaround for that owner
action.

## 2. Capital.com Demo

Capital.com fills the crypto/FX/CFD observation gap while OANDA remains
blocked. Create a Demo account, enable 2FA, then use
**Settings > API integrations > Generate new key**. Save the key when shown
and define a separate custom API-key password.

Then:

```bash
cd ~/Documents/GitHub/lts
./examples/scripts/configure_capital_demo.py
./examples/scripts/enable_capital_demo_observer.sh
```

Verify:

```bash
systemctl --user status lts-capital-demo-observer.timer
python -m app.capital_demo_cli \
  --config examples/configs/capital_demo_execution_lab_v1.json report
```

The code targets the official Demo host and permits only session
authentication plus allowlisted GET requests. Its broker plugin rejects every
order mutation.

## 3. Already Running

No owner action is required for:

- Alpaca Paper account/capability/crypto quote observation every five minutes;
- IBKR Paper capability and quote observation every five minutes while TWS
  has upstream connectivity;
- the USD 100,000 no-order multi-venue shadow portfolio every five minutes;
- deterministic Telegram health and exposure monitoring;
- Hermes paper-business review every 12 hours;
- Moltbook public collection every 30 minutes;
- Moltbook low-cost triage every two hours and supervising Telegram review
  every six hours.

Useful status commands:

```bash
systemctl --user list-timers --all | \
  grep -E 'alpaca|ibkr|shadow|capital|social|watchdog'
cd ~/Documents/GitHub/lts
python -m app.multi_venue_shadow_cli \
  --config examples/configs/multi_venue_shadow_portfolio_v1.json report
```

## 4. MT5/OANDA

The MT5 VM and Linux bridge remain commissioned infrastructure, but an
authenticated OANDA demo session is still blocked by the official
account/support path. Do not enter MetaQuotes community credentials as an
OANDA broker account. Continue only after OANDA supplies the exact demo login,
password and server.
