import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from native_lib import *
spec = execution_specs('confirmation', 'correlated', False)[0]
pol = policies(FRONT['rows'], REPS)
w = CrossedWorld(spec, CONFIG); t = time.time()
r, e = run_capture(w, *pol[REPS[0]]); print('P1 one world', time.time()-t, r['return_pct'], replay('correlated', REPS[0])['worlds'][0] == r)
print(e.trades[0])
