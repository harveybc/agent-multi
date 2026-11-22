"""Write a terminal report for the detached service, including abnormal exits."""
import json
from pathlib import Path
import subprocess
import time

out=Path('/var/tmp/role413-pair-rank-v1')
def read(name):
    return json.loads((out/name).read_text())


def main():
    while not (out/'DONE.json').is_file():
        state=subprocess.run(['systemctl','--user','is-active','role413-pair-rank.service'],
                             capture_output=True,text=True).stdout.strip()
        if state not in ('active','activating','reloading'):
            log=subprocess.run(['journalctl','--user','-u','role413-pair-rank','-n','25','--no-pager'],
                               capture_output=True,text=True).stdout
            (out/'FAILURE.txt').write_text(f'Service: {state}\n{log}')
            return
        time.sleep(15)
    lines=['# 413 TRAIN-only terminal report','',f"Decision: {read('DONE.json')['decision']}",'',
           '| Score year / training | Seed | Arm | Net % | Trades | Trades/week | Offer bp |',
           '|---|---:|---|---:|---:|---:|---:|']
    for tag in ('short','historical','extended'):
        for seed in (7,19,43):
            row=read(f'{tag}-{seed}.json')
            for arm,a in row['arms'].items():
                bp='n/a' if a['offer_bp'] is None else f"{a['offer_bp']:.2f}"
                lines.append(f"| {row['score_year']} / {row['train']} | {seed} | {arm} | {a['net_pct']:.2f} | {a['trades']} | {a['per_week']:.2f} | {bp} |")
    temp=out/'REPORT.tmp'
    temp.write_text('\n'.join(lines)+'\n')
    temp.replace(out/'REPORT.md')


if __name__=='__main__':
    main()
