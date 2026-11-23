"""Attribute the semester-7 switch using archived policies and the native validation ledger."""
import gzip
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role448_worker import Sell, attach_engine, bids, emit, file_sha, heads


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path('/var/tmp/role448-temporal-4plus1')
    run=Path('/var/tmp/role451-activity/run-7')
    read=lambda p:json.loads(p.read_text())
    manifest=root/'7.json'
    spec=read(manifest)
    for name,key in [('7.npz','inputs'),('7.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/name)==spec[key]
    assert torch.cuda.get_device_name()==read(run/'PREFLIGHT.json')['device']
    attach_engine(root/'engine')
    with gzip.open(root/'7.sim.gz','rb') as f:panel,ref=cloudpickle.load(f)[2024]
    assert panel.account(panel.raw)==ref
    with np.load(root/'7.npz') as z:
        x=torch.from_numpy(z['x8']).cuda()
        ix=z['i8']
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    preds,refs,accounts={},{},{}
    for s in [5,6]:
        checkpoint=torch.load(run/f's{s}-validation.pt',map_location='cpu',weights_only=False)
        assert checkpoint['sig']==file_sha(manifest)
        net.load_state_dict(checkpoint['champ']['weights'])
        preds[s]=torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
        refs[s]=checkpoint['champ']['reference']
    for w in [5,6]:
        for c in [5,6]:
            scores=np.zeros_like(panel.raw)
            scores[ix,1,:]=bids(preds[w],refs[c])
            a=panel.account(scores)
            assert a['buy']==0
            if w==c:assert a==read(run/f's{w}-validation.json')['val']
            accounts[f'weights_{w+1}_table_{c+1}']=a
    emit(run/'CROSS_6_7.json',accounts)


if __name__=='__main__':main()
