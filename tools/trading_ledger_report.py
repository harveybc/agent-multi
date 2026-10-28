"""Read authenticated native trades into directional, cost and exit CSV reports.

No fitting, simulation, metric selection or change to a source result. Monetary
aggregates are per run, never a portfolio made by pooling simulation seeds.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def percent(n,d):
    return 100*n/d if d else None


def summarize(trades):
    net=[float(t['net_usd']) for t in trades]
    winners=[v for v in net if v>0];losers=[v for v in net if v<0]
    durations=[(datetime.fromisoformat(t['exit_time'])-datetime.fromisoformat(t['entry_time'])).total_seconds()/3600 for t in trades]
    streak_win=streak_loss=max_win=max_loss=0
    for value in net:
        streak_win=streak_win+1 if value>0 else 0
        streak_loss=streak_loss+1 if value<0 else 0
        max_win=max(max_win,streak_win);max_loss=max(max_loss,streak_loss)
    r=dict(trades=len(trades),wins=len(winners),losses=len(losers),breakeven=sum(v==0 for v in net),
        win_pct=percent(len(winners),len(trades)),net_profit_usd=sum(net),
        gross_profit_usd=sum(t['gross_usd'] for t in trades),
        winning_net_usd=sum(winners),losing_net_usd=sum(losers),
        mean_win_usd=statistics.mean(winners) if winners else None,
        mean_loss_usd=statistics.mean(losers) if losers else None,
        best_trade_usd=max(net) if net else None,worst_trade_usd=min(net) if net else None,
        expectancy_usd=statistics.mean(net) if net else None,
        profit_factor=sum(winners)/-sum(losers) if losers else None,
        median_duration_hours=statistics.median(durations) if durations else None,
        mean_duration_hours=statistics.mean(durations) if durations else None,
        max_duration_hours=max(durations) if durations else None,
        at_most_one_hour_pct=percent(sum(h<=1 for h in durations),len(trades)),
        longest_win_streak=max_win,longest_loss_streak=max_loss,
        gross_positive_net_nonpositive=sum(t['gross_usd']>0 and t['net_usd']<=0 for t in trades),
        ambiguous_intrabar=sum(bool(t['ambiguous']) for t in trades))
    for field in ('spread_usd','slippage_usd','commission_usd','financing_usd'):
        r[field]=sum(t[field] for t in trades)
    r['costs_net_usd']=sum(r[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))
    r['mean_cost_usd']=r['costs_net_usd']/len(trades) if trades else None
    r['payoff_ratio']=r['mean_win_usd']/-r['mean_loss_usd'] if winners and losers else None
    for field in ('units','margin_usd','percentual_order_size'):
        values=[float(t[field]) for t in trades if field in t]
        for suffix,fun in (('min',min),('median',statistics.median),('max',max)):
            r[field+'_'+suffix]=fun(values) if values else None
    sl=[abs(t['entry_price']-t['stop_loss'])*10000 for t in trades]
    tp=[abs(t['take_profit']-t['entry_price'])*10000 for t in trades]
    r['median_sl_pips_eurusd']=statistics.median(sl) if sl else None
    r['median_tp_pips_eurusd']=statistics.median(tp) if tp else None
    r['median_initial_tp_sl_ratio']=statistics.median([b/a for a,b in zip(sl,tp) if a>0]) if sl and all(a>0 for a in sl) else None
    return r


def ledger_checks(trades,report):
    issues=Counter();previous_exit=None
    for t in trades:
        direction=t['direction'];entry=datetime.fromisoformat(t['entry_time']);exit=datetime.fromisoformat(t['exit_time'])
        if direction not in (-1,1):issues['invalid_direction']+=1
        fields=('entry_price','exit_price','units','stop_loss','take_profit','net_usd','gross_usd',
                'entry_cost_usd','exit_cost_usd','spread_usd','slippage_usd','commission_usd','financing_usd')
        if not all(math.isfinite(t[k]) for k in fields):issues['nonfinite']+=1
        if not (t['units']>0 and t['units']==int(t['units'])):issues['invalid_units']+=1
        if not (exit>=entry):issues['negative_duration']+=1
        if previous_exit is not None and entry<previous_exit:issues['overlapping_positions']+=1
        previous_exit=exit
        if not (t['stop_loss']<t['entry_price']<t['take_profit'] if direction==1
                else t['take_profit']<t['entry_price']<t['stop_loss']):issues['invalid_initial_protection']+=1
        gross=direction*t['units']*(t['exit_price']-t['entry_price'])
        costs=sum(t[k] for k in ('spread_usd','slippage_usd','commission_usd','financing_usd'))
        if abs(gross-t['gross_usd'])>1e-8:issues['gross_mismatch']+=1
        if abs(t['gross_usd']-costs-t['net_usd'])>1e-8:issues['net_cost_mismatch']+=1
        if abs(costs-t['entry_cost_usd']-t['exit_cost_usd'])>1e-8:issues['cost_side_mismatch']+=1
    if len(trades)!=report['trades']:issues['native_count_mismatch']+=1
    if abs(sum(t['net_usd'] for t in trades)-report['net_profit_usd'])>1e-7:issues['native_profit_mismatch']+=1
    if trades and abs(sum(t['net_usd']>0 for t in trades)/len(trades)-report['win_rate'])>1e-10:
        issues['native_winrate_mismatch']+=1
    return dict(issues)


def export(roots,output):
    totals=[];directions=[];exits=[];checks=[];sources=[]
    for root in roots:
        for path in sorted(root.rglob('RESULT.json')):
            result=read(path)
            if result.get('state')!='COMPLETE' or 'reports' not in result:continue
            for name,sha in result['artifacts'].items():
                p=(path.parent/name).resolve()
                require(p.is_relative_to(path.parent.resolve()) and digest(p)==sha,'ARTIFACT:'+name)
            for report in result['reports']:
                period=report['period'];require(period in ('TRAIN','INNER_A','OBS_2024'),'PERIOD')
                require(read(path.parent/(period+'.json'))==report,'NATIVE_REPORT_BINDING')
                trades_path=path.parent/(period+'.trades.json');trades=read(trades_path)
                orders=read(path.parent/(period+'.orders.json'))
                ident=dict(campaign=root.name,unit=result.get('unit',report.get('unit',path.parent.name)),
                    period=period,start=report['period_limits'][0],end_exclusive=report['period_limits'][1],
                    training_seed=report.get('seed',report.get('training_seed')),
                    inference_mode=report.get('inference_mode','ARGMAX'),evaluation_seed=report.get('evaluation_seed'))
                row={**ident,**summarize(trades)}
                native=('initial_capital','total_return_pct','sharpe_daily_sqrt365','sortino_daily_sqrt365',
                    'max_drawdown_pct','max_drawdown_usd','orders_per_week','max_entries_per_day',
                    'valid_weekly_rule','weeks_with_decisions','weeks_below_min_orders','weeks_below_min_share',
                    'exposure_bar_share','n_decisions','n_levels_clipped','ruined')
                row.update({k:report.get(k) for k in native})
                row['ending_balance_usd']=report['initial_capital']+row['net_profit_usd']
                row['buy_pct']=percent(sum(t['direction']==1 for t in trades),len(trades))
                row['sell_pct']=percent(sum(t['direction']==-1 for t in trades),len(trades))
                row['voluntary_close_pct']=percent(sum(t['close_reason']=='AGENT_CLOSE' for t in trades),len(trades))
                row.update({'action_'+k:v for k,v in orders['counts'].items()})
                totals.append(row)
                for d,label in ((1,'BUY'),(-1,'SELL')):
                    group=[t for t in trades if t['direction']==d]
                    directions.append({**ident,'direction':label,'share_of_trades_pct':percent(len(group),len(trades)),**summarize(group)})
                for reason in ('AGENT_CLOSE','TP','SL','FORCED_FLAT','REVERSAL'):
                    for d,label in ((1,'BUY'),(-1,'SELL')):
                        group=[t for t in trades if t['direction']==d and t['close_reason']==reason]
                        exits.append({**ident,'direction':label,'close_reason':reason,**summarize(group)})
                issues=ledger_checks(trades,report)
                checks.append({**ident,'ledger_pass':not issues,'issues':json.dumps(issues,sort_keys=True),
                    'scope':'arithmetic, counts, finite fields, protections and single-position chronology; not independent market/calendar validation'})
                sources.append({**ident,'receipt_sha256':digest(path),'trades_sha256':digest(trades_path),
                                'relative_receipt':str(path.relative_to(root))})
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('runs',totals),('directions',directions),('exits',exits),('checks',checks)]:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    metadata=dict(schema='trading_ledger_report.v1',generated_at=datetime.now().astimezone().isoformat(),
        reports=len(totals),failed_ledger_checks=sum(not r['ledger_pass'] for r in checks),
        definitions=dict(order='one opened and subsequently closed trade, not two orders or an action intent',
            win='net_usd > 0 after spread/slippage/commission/financing; zero is separate',
            directional_win_rate='winning trades of that direction / all trades of that direction',
            empty_direction='count zero, win rate undefined (empty CSV cell)',
            financing='net debit, negative means credit',
            duplication='Original CAT_FIXED and ARGMAX inference rows replay the same evidence; do not pool as independent samples',
            scopes='TRAIN and INNER are distinct; OBS_2024 is prior stored observation, no new evaluation'),
        sources=sources,artifacts={p.name:digest(p) for p in output.glob('*.csv')})
    (output/'MANIFEST.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ('reports','failed_ledger_checks','generated_at')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    require(all(r.exists() and not a.output.resolve().is_relative_to(r.resolve()) for r in a.root),'OUTPUT_ISOLATION')
    export(a.root,a.output)
