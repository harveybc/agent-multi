"""Task C: spot-check doc 126 table rows against PROFILES.json (positive_calendar population)."""
import hashlib, json, sys
p = sys.argv[1]
raw = open(p, 'rb').read()
print('PROFILES sha256', hashlib.sha256(raw).hexdigest())
d = json.loads(raw)
print('schema', d.get('schema'), 'train', d.get('train'), 'source_identity', d.get('source_identity'))
print('top keys', sorted(d.keys()))
# doc 126 rows: (h, side, n, tp_mean, tp_std, tp_median, mae_mean, mae_std, mae_median, dur_median, dur_p90)
doc126 = {
    (3, 'buy'): (15033, 28.17, 26.03, 20.0, 11.24, 12.16, 7.4, 5, 12),
    (3, 'sell'): (15053, 28.06, 27.73, 19.6, 12.03, 14.01, 7.7, 5, 12),
    (6, 'buy'): (14946, 33.47, 30.45, 24.0, 13.05, 14.20, 8.5, 7, 16),
    (6, 'sell'): (14845, 32.75, 30.85, 23.4, 13.53, 14.80, 8.9, 7, 16),
    (12, 'buy'): (12932, 46.52, 37.71, 35.3, 17.64, 17.94, 11.9, 12, 27),
    (12, 'sell'): (13191, 47.86, 40.95, 35.8, 19.22, 20.48, 12.7, 12, 29),
    (24, 'buy'): (10284, 53.90, 41.67, 42.1, 22.06, 20.89, 16.0, 17, 41),
    (24, 'sell'): (10765, 54.52, 44.37, 43.1, 23.51, 24.04, 16.2, 17, 42),
}
doc126_support = {3: (19620, 259, 696, 824, 256, 253), 6: (19108, 771, 796, 1041, 221, 221),
                  12: (16191, 3688, 581, 768, 144, 144), 24: (12539, 7340, 364, 363, 95, 100)}
ok = True
for row in d['rows']:
    h = row['horizon']
    print(f"h={h} params={row['params']} support={row.get('support')}")
    s = row['support']
    exp = doc126_support[h]
    pb = row['directions']['buy']['positive_excluded_by_calendar']
    ps = row['directions']['sell']['positive_excluded_by_calendar']
    zb = row['directions']['buy']['positive_calendar']['zero_mae']
    zs = row['directions']['sell']['positive_calendar']['zero_mae']
    got = (s['complete'], s['censored'], pb, ps, zb, zs)
    flag = 'OK' if got == exp else 'MISMATCH'
    ok &= got == exp
    print(f"  support table doc126={exp} json={got} {flag}")
    for side in ('buy', 'sell'):
        pc = row['directions'][side]['positive_calendar']
        got = (pc['n'], round(pc['tp_pips']['mean'], 2), round(pc['tp_pips']['std'], 2), round(pc['tp_pips']['median'], 1),
               round(pc['mae_pips']['mean'], 2), round(pc['mae_pips']['std'], 2), round(pc['mae_pips']['median'], 1),
               pc['duration_hours']['median'], pc['duration_hours']['p90'])
        exp = doc126[(h, side)]
        match = all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(got, exp))
        ok &= match
        print(f"  {side}: doc126={exp}\n        json  ={got} {'OK' if match else 'MISMATCH'}")
print('ALL_ROWS_MATCH' if ok else 'SOME_MISMATCH')
