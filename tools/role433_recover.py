"""Replay only required stages in an isolated tree; authenticate against 432."""
import hashlib
import sys
from pathlib import Path
import torch
from role432_guard import run,w


def main(src,out,tag):
    path=src/f'{tag}.npz'
    sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    for f in sorted(src.glob(tag+'.s*.pt')):
        d=torch.load(f,map_location='cpu',weights_only=False)
        wanted=[r for r in d['repairs'] if r[0]==1 and r[-1]]
        if not wanted:continue
        stage=int(f.suffixes[-2][2:])
        work=out/(f.stem+'-replay');work.mkdir(parents=True,exist_ok=True)
        p=work/path.name
        if not p.exists():p.symlink_to(path)
        for i in range(1,stage):
            q=p.with_suffix(f'.s{i}.pt')
            if not q.exists():q.symlink_to(src/q.name)
        target=p.with_suffix(f'.s{stage}.pt')
        if target.exists():target.unlink()
        cap={}
        def capture(s,r,accepted):
            assert s==stage and r[0]==wanted[0][2] and r[3]==wanted[0][1]
            cap.update(raw=r[1],accepted=accepted,parent=d['parent'][1])
        run(p,sha,last_stage=stage,capture=capture)
        replay=torch.load(target,map_location='cpu',weights_only=False)
        assert replay['repairs']==d['repairs'] and replay['history']==d['history']
        for k,v in d['champs'][1].items():assert torch.equal(v,replay['champs'][1][k])
        w.save(dict(**cap,wanted=wanted,adopted=d['adopted']),out/f.name)


if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
