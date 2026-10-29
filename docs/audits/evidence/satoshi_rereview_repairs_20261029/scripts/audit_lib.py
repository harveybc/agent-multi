"""Auditor's independent re-implementation (no import of protection_reference or controller math).

Rules re-derived from heuristic-strategy e0a1c2b env.py/_protect/_close:
  * exit quote = mid - direction*spread/2 (long exits on bid, short exits on ask), spread of the bar OPEN;
  * at an open: SL (o beyond SL, inclusive) then TP (o beyond TP, inclusive) fill AT THE OPEN QUOTE;
  * else an agent CLOSE fills at the open quote (market slippage);
  * else intrabar: adverse extreme touching SL -> fill at SL (stop slippage), stop wins ambiguity;
    favourable extreme beyond TP by limit_through -> fill at TP (no slippage);
  * forced flat: close of the last bar before the deadline (market slippage);
  * financing: rollover events with entry_time < t <= exit_ns; exit_ns = bar open for open fills,
    bar open + 1h for intrabar and close fills.
"""
import math


def independent_values(direction, entry, units, sl, tp, entry_slip, market_slip, stop_slip, commission_rt,
                       through, paths, spreads, fund_open, fund_end):
    """Return per-sample list of H+1 net values: close at open j (j<H) then passive."""
    out = []
    H = len(paths[0])
    for path, spr in zip(paths, spreads):
        q = [[float(v) - direction * float(s) / 2 for v in bar] for bar, s in zip(path, spr)]
        passive = None
        last_close = None  # last index at which a discretionary open close is still executable
        for k in range(H):
            o, h, l, c = q[k]
            if direction * (o - sl) <= 0:
                passive = units * (direction * (o - entry) - entry_slip - stop_slip - commission_rt - fund_open[k])
                last_close = k - 1
                break
            if direction * (o - tp) >= 0:
                passive = units * (direction * (o - entry) - entry_slip - 0. - commission_rt - fund_open[k])
                last_close = k - 1
                break
            adverse = l if direction == 1 else h
            favourable = h if direction == 1 else l
            if direction * (adverse - sl) <= 0:
                passive = units * (direction * (sl - entry) - entry_slip - stop_slip - commission_rt - fund_end[k])
                last_close = k
                break
            if direction * (favourable - tp) >= through:
                passive = units * (direction * (tp - entry) - entry_slip - commission_rt - fund_end[k])
                last_close = k
                break
        if passive is None:
            c = q[H - 1][3]
            passive = units * (direction * (c - entry) - entry_slip - market_slip - commission_rt - fund_end[H - 1])
            last_close = H - 1
        row = []
        for j in range(H):
            if j <= last_close:
                o = q[j][0]
                row.append(units * (direction * (o - entry) - entry_slip - market_slip - commission_rt - fund_open[j]))
            else:
                row.append(passive)
        row.append(passive)
        out.append(row)
    return out


def independent_advantage(values, weights, z, risk):
    n = len(values); A = len(values[0])
    best = None; per = []
    for j in range(1, A):
        d = [values[s][0] - values[s][j] for s in range(n)]
        mu = math.fsum(w * x for w, x in zip(weights, d))
        var = math.fsum(w * (x - mu) ** 2 for w, x in zip(weights, d))
        score = mu - z * math.sqrt(var)
        per.append(score)
        best = score if best is None else min(best, score)
    return best / risk, per


class IndependentPersistence:
    def __init__(self, margin, persistence, urgent):
        self.margin, self.persistence, self.urgent = margin, persistence, urgent
        self.pid = None; self.last = None; self.streak = 0

    def step(self, pid, t, step, adv):
        contiguous = self.pid == pid and self.last is not None and t == self.last + step
        s = self.streak if contiguous else 0
        q = adv > max(self.margin, 1e-12)
        self.streak = s + 1 if q else 0
        self.pid, self.last = pid, t
        u = q and self.urgent is not None and adv >= self.urgent
        return dict(qualifies=q, urgent=u, streak=self.streak,
                    action='CLOSE' if q and (u or self.streak >= self.persistence) else 'HOLD')
