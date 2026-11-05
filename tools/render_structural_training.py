"""Generate local PNG/CSV diagnostics without loading weights or evaluating models."""
import argparse
import json
from pathlib import Path

from app.training_diagnostics import render_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    spec=json.loads((args.root/'SPEC.json').read_text())
    run=args.root/'runs'/f'seed_{args.seed}'
    history=json.loads((run/'HISTORY.json').read_text())
    result=render_diagnostics(history,args.output or run/'diagnostics',spec['design'])
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    main()
