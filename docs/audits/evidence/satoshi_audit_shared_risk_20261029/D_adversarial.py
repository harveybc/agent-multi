"""Task D: adversarial fixtures against app.horizon_risk_rules with the pinned native Sizing."""
import math, sys, traceback
from pathlib import Path
ROOT = Path(sys.argv[1]); ENGINE = Path(sys.argv[2])
sys.path.insert(0, str(ROOT))
from app.horizon_risk_rules import coefficients, levels, propose
from app.zigzag_opportunities import engine
_, _, X = engine(ENGINE)
S = X.Sizing(rel_volume=.1, leverage=50, basis='margin')

base = dict(direction=1, entry_quote=1.1, liquidation_quote=1.0999, balance=10000.,
            tp_reference=.0020, mae_reference=.00074, sigma=.0005, genes=coefficients(),
            sizing=S, tick=.00001, min_distance=.00005, quality=1.,
            loss_cost_per_unit=.00002+.00005+2*3.5/100000, funding_reserve_per_unit=0.)

def run(name, **changes):
    args = base | changes
    try:
        out = propose(**args)
        keep = {k: out[k] for k in ('status', 'reason', 'q', 'units', 'sl', 'tp', 'margin', 'estimated_loss', 'risk_budget', 'loss_per_unit') if k in out}
        print(f"[{name}] -> {keep}")
        return out
    except Exception as e:
        print(f"[{name}] -> EXCEPTION {type(e).__name__}: {e}")
        return None

print('== baseline (reference-like h3 buy, median refs, 1 pip spread) ==')
b = run('baseline')
print('   check: rho*balance =', .005*10000, '; units*loss_per_unit =', b['units']*b['loss_per_unit'])
print('   max units q=1 =', S.units_max(10000., 1.1), '; margin at q=1 =', S.required_margin(S.units_max(10000., 1.1), 1.1))

print('\n== (i) tiny balance: integer units round to 0 ==')
for bal in (1.0, 0.5, 0.2, 0.01):
    run(f'balance={bal}', balance=bal)
print('   units_max(balance=0.2, 1.1) =', S.units_max(0.2, 1.1), '; units_max(0.5,1.1)=', S.units_max(.5, 1.1))
print('   q-driven rounding to zero: balance 100, rho .001 -> budget .1 USD; loss_per_unit ~', b['loss_per_unit'])
run('balance=100 rho=.001', balance=100., genes=coefficients(rho=.001))
run('balance=30 rho=.001', balance=30., genes=coefficients(rho=.001))

print('\n== (ii) spread larger than buffer ==')
# buffer 1 pip, MAE 7.4 pips -> SL 8.4 pips below entry; spread 2, 5, 8, 8.4, 9, 20 pips
for sp in (2, 5, 8, 8.3, 8.4, 9, 20):
    run(f'spread={sp}pips buffer=1', liquidation_quote=1.1-sp*.0001)
print('   MAE=0 with buffer 1 pip and spread 0.5 / 1 / 1.5 pips:')
for sp in (.5, .9, .95, 1.0, 1.5):
    run(f'mae=0 spread={sp}pips', mae_reference=0., liquidation_quote=1.1-sp*.0001)
print('   buffer=0 and MAE=0, spread 0 (liquidation == entry):')
run('mae=0 buffer=0 spread=0', mae_reference=0., genes=coefficients(buffer_pips=0.), liquidation_quote=1.1)

print('\n== (iii) negative / NaN / inf sigma ==')
for sg in (-.0001, float('nan'), float('inf'), -0.0):
    run(f'sigma={sg}', sigma=sg)
print('   negative b_sl / b_tp genes, nan rho:')
for g in (dict(b_sl=-.1), dict(b_tp=-.1), dict(rho=float('nan')), dict(rho=1.5), dict(a_sl=0.)):
    try:
        print(f'   coefficients(**{g}) ->', coefficients(**g))
    except Exception as e:
        print(f'   coefficients(**{g}) -> {type(e).__name__}: {e}')
print('   large sigma with b_tp=1 erodes TP below zero distance:')
run('sigma=.003 b_tp=1', sigma=.003, genes=coefficients(b_tp=1.))
print('   large sigma with b_sl=1 widens SL (q must drop):')
run('sigma=.003 b_sl=1', sigma=.003, genes=coefficients(b_sl=1.))

print('\n== rounding direction check ==')
for d in (1, -1):
    lv = levels(d, 1.100003, .0020, .00074, .0005, coefficients(), .00001)
    raw_sl = 1.100003 - d*(.00074+.0001); raw_tp = 1.100003 + d*.0020
    print(f'   dir={d} raw_sl={raw_sl:.6f} sl={lv["sl"]:.6f} (dist {d*(1.100003-lv["sl"])*1e4:.3f} pips vs raw {lv["raw_sl_distance"]*1e4:.3f}) raw_tp={raw_tp:.6f} tp={lv["tp"]:.6f} (dist {d*(lv["tp"]-1.100003)*1e4:.3f} pips vs raw {lv["raw_tp_distance"]*1e4:.3f})')

print('\n== budget claim: loss modeled vs. gap; stop slippage already in loss_cost ==')
o = run('budget probe rho=.005')
gap = 1.1 - .01
hit = X._protect(1, o['tp'], o['sl'], gap, gap, gap)
print('   native _protect on 100-pip adverse gap ->', hit, '; realized loss =', o['units']*((1.1-hit[1])+base['loss_cost_per_unit']), 'vs budget', o['risk_budget'])
print('   utilization =', o['estimated_loss']/o['risk_budget'])
# Is the cap guaranteed? loss = units*loss_per_unit, units = floor(q*max) <= q*max, q <= budget/(max*loss_per_unit) -> loss <= budget.
print('   algebra: units=floor(q*umax) <= q*umax = min(1,budget/(umax*lpu))*umax -> loss=units*lpu <= budget. Holds iff loss linear in units and SL fill at SL (no gap), stop slippage fixed per unit.')
# Rounding of SL away from entry increases loss_per_unit vs raw -- is the rounded distance used? Check:
print('   loss_per_unit uses rounded SL:', o['loss_per_unit'], '= rounded dist', (1.1-o['sl']), '+ cost', base['loss_cost_per_unit'])
print('   native order_units(min rounding) with risk_accepts_minimum=False: units for q tiny ->', S.order_units(10000., 1.1, percentual_order_size=1e-6, risk_accepts_minimum=False))
