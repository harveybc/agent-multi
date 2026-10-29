import gzip, json, sys, collections
c = json.loads(gzip.decompress(open(sys.argv[1], 'rb').read())); ref = {r['spec']['id']: r for r in json.loads(gzip.decompress(open(sys.argv[2], 'rb').read()))}
rows = [r for r in c if r['arm'] == 'prefix_guarded']
out = collections.Counter(); where = collections.Counter(); modes = collections.Counter(); anyexec = 0
for r in rows:
    R = ref[r['spec']['id']]; tr = [t for t in r['trace'] if 'advantage_risk' in t]
    pos_req = [t for t in tr if t['advantage_risk'] > 0.02]
    pos_z0 = [t for t in tr if t['expected_close_net'] - t['best_expected_wait_net'] > 1e-7]
    out['early_close'] += r['metrics']['early_close']
    anyexec += any(t['decision_bar'] + 1 in R['fill_bars'] for t in pos_req + pos_z0)
    if pos_req:
        f = pos_req[0]; out['first_positive'] += 1
        where['fill_bar_minus_passive_exit_bar=%d' % (f['decision_bar'] + 1 - R['passive_exit_bar'])] += 1
        modes[(r['spec']['mode'], r['native_reason'])] += 1
    else:
        out['none'] += 1; modes[('NONE', r['spec']['mode'])] += 1
    out['ticks'] += len(tr); out['ticks_qualifying'] += sum(t['qualifies'] for t in tr)
print(json.dumps(dict(counts=out, any_executable_request_any_tick=anyexec, first_request_position=where,
                      modes={str(k): v for k, v in modes.items()}), indent=1))
