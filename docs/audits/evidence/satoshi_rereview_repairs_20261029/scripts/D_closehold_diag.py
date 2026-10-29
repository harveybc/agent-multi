"""Diagnose the 2.8e-14 expected_close_net differences: compare patched vs pre-repair value_paths bitwise on the
same reports, and the dot product w@values[:,0] computed in different memory layouts."""
import gzip, json, sys, types
from pathlib import Path
AM, HS, PRE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); sys.path.insert(0, str(AM))
import numpy as np
from app.zigzag_opportunities import engine; engine(HS)
import app.close_hold_controller as C
from app.close_hold_verification import arms, run_case
# pre-repair value_paths + protection_exit loaded from source text into a private namespace
old_pe = types.ModuleType('old_pe'); exec((PRE/'app/protection_reference.py').read_text(), old_pe.__dict__)
old_c = types.ModuleType('old_c'); src = (PRE/'app/close_hold_controller.py').read_text().replace(
    'from app.protection_reference import protection_exit', '')
old_c.__dict__['protection_exit'] = old_pe.protection_exit; exec(src, old_c.__dict__)
root = AM/'docs/audits/evidence/absolute_early_close_20261029'
refs = json.loads(gzip.decompress((root/'CASES.json.gz').read_bytes()))
config = json.loads((AM/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())
old = json.loads(gzip.decompress((AM/'docs/audits/evidence/close_hold_controller_20261029/CASES.json.gz').read_bytes()))
log = []
orig_vp = C.value_paths
def spy(p, r, fo, fe):
    v = orig_vp(p, r, fo, fe)
    shim = old_c.PathReport(r.issued_at, r.observed_through, r.first_fill, r.step, r.deadline, r.paths, r.spreads, r.weights, r.provenance)
    vo = old_c.value_paths(p, shim, fo, fe)
    w = np.asarray(r.weights, float)
    log.append(dict(values_bitwise_equal=bool(np.array_equal(v, vo)), max_abs=float(np.max(np.abs(v-vo))),
                    terminal_eq_last_bar_spread=bool(np.all(r.terminal_spreads == r.spreads[:, -1])),
                    dot_strided=float(w@v[:, 0]), dot_contig=float(w@np.ascontiguousarray(v[:, 0])),
                    dot_old=float(w@vo[:, 0]), sum3=float(w[0]*v[0, 0]+w[1]*v[1, 0]+w[2]*v[2, 0]) if len(w) == 3 else None,
                    v_aligned16=v.ctypes.data % 16, v_aligned32=v.ctypes.data % 32, v_aligned64=v.ctypes.data % 64))
    return v
C.value_paths = spy
out = []
arm_list = arms()
for idx in (122, 123, 127, 128, 132):
    spec = old[idx]['spec']; arm = old[idx]['arm']
    ref = next(r for r in refs if r['spec'] == spec)
    log.clear()
    arm_spec = next(a for a in arm_list if a['name'] == arm)
    rec = json.loads(json.dumps(run_case(spec, config, ref, arm_spec), sort_keys=True))
    diffs = [(i, a['expected_close_net'], b['expected_close_net']) for i, (a, b) in enumerate(zip(rec['trace'], old[idx]['trace'])) if a != b]
    out.append(dict(idx=idx, arm=arm, ticks=len(log), differing_ticks=[d[0] for d in diffs][:6],
                    values_all_bitwise_equal=all(l['values_bitwise_equal'] for l in log),
                    max_abs_values=max(l['max_abs'] for l in log),
                    terminal_eq_bar_spread_all=all(l['terminal_eq_last_bar_spread'] for l in log),
                    at_diff=[dict(tick=d[0], new=d[1], committed=d[2], **{k: log[d[0]][k] for k in ('dot_strided', 'dot_contig', 'dot_old', 'sum3', 'v_aligned16', 'v_aligned32', 'v_aligned64')}) for d in diffs[:3]]))
print(json.dumps(out, indent=1, default=str))
