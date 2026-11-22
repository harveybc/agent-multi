import unittest

import numpy as np
import pandas as pd
from tools.scenario393 import compose, context, dynamics


class MacroContract(unittest.TestCase):
    def rows(self):
        s = np.repeat([0, 1, 0, 1, 0], [4, 5, 3, 6, 4])
        i = np.arange(len(s))
        return pd.DataFrame(dict(state=s,start=50+2*i,end=52+2*i,n=2,
                                 side=1-2*(i%2),start_ns=i,displacement=(1-2*(i%2))*.001,
                                 vol=.002,retrace=.0001,retrace_at=.5,curvature=0.,
                                 first=.02,middle=.02,last=.02))

    def test_durations_exclude_both_censored_ends(self):
        _,dur,edges=dynamics(self.rows())
        np.testing.assert_array_equal(dur[0],[3])
        np.testing.assert_array_equal(dur[1],[5,6])
        self.assertEqual(set(edges[1]),{0})

    def test_future_does_not_change_context(self):
        rows=self.rows().iloc[:1]
        price=1+np.arange(110)**1.5/1e5
        before=context(rows,price)
        price[51:]*=3
        np.testing.assert_array_equal(before,context(rows,price))

    def test_schedule_exact_and_repeatable_with_terminal_censor(self):
        clock=pd.date_range("2016-01-01",periods=22,tz="UTC",freq="h").append(
              pd.DatetimeIndex(["2019-12-31T20:00:00Z"]))
        r=self.rows()
        a=compose(r,clock,19)
        pd.testing.assert_frame_equal(a,compose(r,clock,19))
        self.assertEqual(a.bars.sum(),len(clock))
        self.assertTrue(a.iloc[-1].censored)
        np.testing.assert_array_equal(a.side.iloc[1:],-a.side.iloc[:-1])
        self.assertEqual(a.reopenings.sum(),1)
        with self.assertRaises(ValueError):
            compose(r,clock.append(clock[-1:]),19)
