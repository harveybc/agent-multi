"""D2: short-side limit_through boundary, native two-position fixture, corpus-sample full-menu check."""
import gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from fixtures_lib import *  # noqa: F401,F403  (helpers only; AM_ROOT/HS_ROOT from the environment)
from app.absolute_exit_corpus import create_case

rows = []
# F6s short TP intrabar exactly at limit_through boundary (ask low = TP - through).
for tag, extra in (('at', 0.), ('below', 1e-6)):
    e0, _ = make_env(-1, 10, 10, 12, [])
    q = e0.pos.tp - 1e-4 - extra - 3e-5 / 2  # ask low = mid + spread/2
    e, risk = make_env(-1, 10, 10, 12, [(1.0998, 1.0999, q, 1.0998)] if False else
                       [(1.0998, 1.0999, 1.0997, 1.0998), (1.0996, 1.0997, q, 1.0996)])
    e.step(e.next_decision, 0)
    rows.append(evaluate(f'F6s_short_TP_through_{tag}', e, risk, 'limit_through boundary, short'))
# F7 native two-position witness: same future market, tight vs wide stop.
fut = [(1.1001, 1.1002, 1.1000, 1.1001), (1.1004, 1.1005, 1.0985, 1.0987), (1.0987, 1.1030, 1.0987, 1.1029)]
for tag, sl, tp in (('tight', 10, 30), ('wide', 25, 25)):
    e, risk = make_env(1, sl, tp, 12, fut)
    e.step(e.next_decision, 0)
    rows.append(evaluate(f'F7_two_positions_{tag}', e, risk, 'same market, different stop'))
# Corpus sample: every 9th of the 1744 positions, at the first decision tick: every common action vs native.
config = json.loads(Path(AM, 'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
from app.absolute_exit_corpus import specs
sample = list(specs())[::9]
worst = 0.; worst_id = None; n_actions = 0
for spec in sample:
    e, risk = create_case(spec, config)
    r = evaluate(spec['id'], e, risk, 'corpus')
    n_actions += r['H'] + 1
    if r['max_abs_ctrl_vs_native'] > worst:
        worst, worst_id = r['max_abs_ctrl_vs_native'], spec['id']
rows.append(dict(corpus_sample_positions=len(sample), actions_compared=n_actions,
                 max_abs_ctrl_vs_native_usd=worst, at=worst_id))
print(json.dumps(rows, indent=1, default=float))
