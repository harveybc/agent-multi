"""Collect completed remote folds and run each paired account once it is ready."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def collect(a):
    root=a.root.resolve()
    for _ in range(2880):  # 24h time limit; rerunning the collector is idempotent.
        for year in range(2020,2024):
            target=root/str(year)
            if (target/'accounts.json').exists():
                continue
            if year!=2022:
                cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,
                     'test','-f',str(target/'long/done.json')]
                probe=subprocess.run(cmd)
                if probe.returncode in (1,255):
                    continue
                probe.check_returncode()
                for task in ('short','long'):
                    folder=target/task
                    folder.mkdir(exist_ok=True)
                    subprocess.run(['rsync','-a',*[f'{a.host}:{folder}/{f}' for f in
                        ('done.json','pred.npz','champ.pt','status.json')],str(folder)+'/'],check=True)
            if all((target/k/'done.json').exists() for k in ('short','long')):
                subprocess.run([sys.executable,'-m','tools.role477_accounts',str(target),
                    str(a.history),str(a.market),str(a.bundle)],check=True)
                print(year,(target/'accounts.json').read_text(),flush=True)
        if all((root/str(y)/'accounts.json').exists() for y in range(2020,2024)):
            result={y:json.loads((root/str(y)/'accounts.json').read_text()) for y in range(2020,2024)}
            tmp=root/'result.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n')
            tmp.replace(root/'RESULT.json')
            return
        time.sleep(30)
    raise TimeoutError('fold collection deadline; inspect worker status before resuming')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','history','market','bundle'):p.add_argument(k,type=Path)
    p.add_argument('host')
    collect(p.parse_args())
