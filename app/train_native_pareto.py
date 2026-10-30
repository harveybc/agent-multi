"""Restricted TRAIN-prefix oracle with explicit BID quotes and native execution."""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from app.dynamic_account_panel import World, HOUR
from app.sequential_account_pareto import clone, enumerate_tree, finish, nondominated


def bid_environment_class():
    from app.short_horizon.expert_guided.absolute import AbsoluteBarEnv

    class BidAbsoluteBarEnv(AbsoluteBarEnv):
        def quote_price(self, k, bid, quote_side, time_ns=None):
            if quote_side not in (-1,1):
                raise ValueError('QUOTE_SIDE')
            when = int(self.v[k]) if time_ns is None else int(time_ns)
            spread = self.cfg.profile.spread_at(when,int(self.mo[int(self.seg_of[k])]))
            if not math.isfinite(spread) or spread < 0:
                raise ValueError('QUOTE_SPREAD')
            return float(bid) + (spread if quote_side == 1 else 0.)

    return BidAbsoluteBarEnv


def calendar_sample(bars, calendar):
    """First complete session in each TRAIN year; no return or pivot inputs."""
    selected = {}
    times = bars.index.as_unit('ns').asi8
    for seg in calendar.segments:
        year = seg.market_open.year
        if year not in (2020,2021,2022,2023) or year in selected:
            continue
        a,b = np.searchsorted(times,[seg.market_open.value,seg.market_close.value])
        if b <= a or times[a] != seg.market_open.value or times[b-1]+HOUR != seg.market_close.value:
            continue
        if np.any(np.diff(times[a:b]) != HOUR):
            continue
        selected[year] = (int(a),int(b),seg)
    if set(selected) != {2020,2021,2022,2023}:
        raise ValueError('FOUR_COMPLETE_TRAIN_SESSIONS_REQUIRED')
    return selected


@dataclass
class Node:
    env: object
    equity: list
    planned_exit: int | None
    plans: list


class TrainPrefixWorld(World):
    def __init__(self, bars, segment, global_start, graphs, risk, profile):
        from app.short_horizon import calendar as CAL, execution as X
        from app.short_horizon.expert_guided.env import EnvConfig
        cal=CAL.load_calendar(dict(calendar_id='TRAIN_EXACT_PREFIX',version='1',
            segments=[dict(segment_id='0',market_open=segment.market_open.isoformat(),market_close=segment.market_close.isoformat())],
            entry_delay_after_open_hours=12,force_flat_before_close_hours=12,entry_cutoff_before_flat_hours=0))
        cfg=EnvConfig(profile=profile,sizing=X.Sizing(rel_volume=.1,leverage=50),
                      execution_price_mode='bid_ask_v3',max_entries_per_day=None,min_orders_per_week=2.)
        self.env=bid_environment_class()(bars,cal,np.full(len(bars),.001),np.full(len(bars),.002),
            bars.index[0],bars.index[-1]+pd.Timedelta(hours=1),cfg,tick=.00001)
        self.end=int(np.searchsorted(self.env.v,self.env.ff[0]))
        self.risk_config=risk
        self.references={(r['horizon'],1 if r['direction']=='buy' else -1):r['reference']['median'] for r in risk['rows']}
        self.rolls=[(r.value,3 if i==profile.triple_swap_rollover_index else 1)
                    for i,r in enumerate(cal.segments[0].rollovers(),1)]
        self.graphs=graphs
        self.origin_rows={h:{int(o):i for i,o in enumerate(g['origin'])} for h,g in graphs.items()}
        self.global_start=global_start

    def initial(self):
        e=clone(self.env)
        e.reset()
        equity=[e.cfg.capital]
        while not e.entry_gate(e.next_decision)[0]:
            b=e.next_decision
            if b >= self.end-2:
                raise ValueError('NO_PREFIX_START')
            e.step(b,0)
            equity.append(float(e.equity_at(b+1)))
        return Node(e,equity,None,[])

    def choices(self, node, allow_entry=True):
        e=node.env
        if e.pos is not None:
            return ['PIVOT_CLOSE'] if e.next_decision >= node.planned_exit else ['HOLD','CLOSE']
        choices=['NOP']
        if not allow_entry or not e.entry_gate(e.next_decision)[0]:
            return choices
        origin=self.global_start+e.next_decision+1
        for h,g in sorted(self.graphs.items()):
            row=self.origin_rows[h].get(origin)
            if row is None or not g['complete'][row]:
                continue
            for k,end in enumerate(g['candidate_end'][row]):
                local=int(end)-self.global_start
                # Pivot bar must finish by weekly flat; no substituted timed exit.
                if not e.next_decision+1 < local < self.end:
                    continue
                for side in ('BUY','SELL'):
                    choices.append(f'{side}:{h}:{k}')
        return choices

    def opening_support(self,node):
        origin=self.global_start+node.env.next_decision+1
        result={}
        for h,g in sorted(self.graphs.items()):
            r=self.origin_rows[h].get(origin)
            if r is None or not g['complete'][r]:
                result[str(h)]=dict(status='NO_ORIGIN' if r is None else 'CENSORED_K3',available_endpoints=[])
            else:
                available=[int(k) for k,end in enumerate(g['candidate_end'][r])
                           if node.env.next_decision+1 < int(end)-self.global_start < self.end]
                result[str(h)]=dict(status='COMPLETE_K3',available_endpoints=available,
                    excluded_by_execution_boundary=3-len(available))
        return result

    def transition(self, node, action):
        if action not in self.choices(node):
            raise ValueError('ILLEGAL_TRAIN_ACTION')
        e=clone(node.env)
        deadline=node.planned_exit
        plans=list(node.plans)
        plan=None
        a=3 if action in ('CLOSE','PIVOT_CLOSE') else 0
        if ':' in action:
            side,h,k=action.split(':'); h,k=int(h),int(k)
            a=1 if side=='BUY' else 2
            origin=self.global_start+e.next_decision+1
            r=self.origin_rows[h][origin]
            deadline=int(self.graphs[h]['candidate_end'][r,k])-self.global_start
            plan=self.entry_plan(h,1 if a==1 else -1,1.)
        b=e.next_decision
        info=e.step(b,a,entry_plan=plan)
        if info['refused']:
            raise AssertionError(('TRAIN_ORDER_REFUSAL',info))
        if info['entry']:
            plans.append(dict(entry=b+1,horizon=h,pivot_slot=k,pivot_bar=deadline))
        return Node(e,node.equity+[float(e.equity_at(b+1))],deadline if e.pos is not None else None,plans)

    def settle(self,node):
        while node.env.next_decision < self.end-1:
            choices=self.choices(node,allow_entry=False)
            action='PIVOT_CLOSE' if choices==['PIVOT_CLOSE'] else ('HOLD' if node.env.pos is not None else 'NOP')
            node=self.transition(node,action)
        result=finish(node.env,node.equity)
        result['plans']=node.plans
        return result

    def exact(self,steps=3,max_nodes=100000):
        root=self.initial()
        branch_end=root.env.next_decision+steps
        if branch_end >= self.end-1:
            raise ValueError('PREFIX_CROSSES_FLAT')
        def terminal(node):
            return self.settle(node)['vector'] if node.env.next_decision>=branch_end else None
        def expand(node):
            return [(a,self.transition(node,a)) for a in self.choices(node)]
        leaves,nodes=enumerate_tree(root,expand,terminal,max_nodes=max_nodes)
        return dict(nodes=nodes,terminal_accounts=len(leaves),frontier=nondominated(leaves),
                    initial_actions=self.choices(root),initial_support=self.opening_support(root),steps=steps),leaves

    def replay(self,trace):
        node=self.initial()
        for a in trace:
            node=self.transition(node,a)
        return self.settle(node)
