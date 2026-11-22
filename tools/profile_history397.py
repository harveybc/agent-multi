"""Bounded, disposable profile of the history397 SAC loop. No production writes."""
import json
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from agent_plugins.discrete_curve_sac import tensor_observation
from agent_plugins.sac_critic_regularization import module_mode
from tools import pilot391 as p
from tools.account397 import assemble
from tools.run_discrete_curve_probe import transition


def main(dest):
    root = Path('/var/tmp/pilot397')
    parent = Path('/var/tmp/sac70-warm-v3-20261108')
    bundle = root / 'model/61/bundle'
    p.configure(7)
    torch.backends.cudnn.allow_tf32 = False
    m, _ = p.load_bundle(bundle, 'cuda')
    train, _ = assemble(parent, bundle, root / 'history397-61-r2')
    d = p.read(parent / 'SPEC.json')['design']
    env = p.environment(train, p.read(parent/'RISK.json'), d['horizon'],
                        ['2016-01-01','2024-01-01'], m.effective['gamma'])
    obs = env.reset()
    bs = d['batch_size']
    out = dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), batch=bs,
               groups=len(m.actor_net.encoder.branches), update_every=d['update_every'],
               obs_shape=list(obs['features'].shape), model=p.file_sha(bundle/'model.pt'),
               params=sum(v.numel() for v in m.loss.parameters()),
               trainable=sum(v.numel() for v in m.loss.parameters() if v.requires_grad))
    clock = time.perf_counter
    sync = torch.cuda.synchronize
    with tempfile.TemporaryDirectory() as tmp:
        replay = TensorDictReplayBuffer(storage=LazyMemmapStorage(d['replay_size'], scratch_dir=tmp),
                                        batch_size=bs)
        for i in range(bs*3):
            nxt, r, done, dt = env.step(i % 3)
            replay.extend(transition(obs,i%3,nxt,r,done,dt))
            obs = nxt

        def block(n, fence):
            nonlocal obs
            sums = defaultdict(float)
            sync()
            start = mark = clock()
            def lap(k):
                nonlocal mark
                if fence: sync()
                t = clock()
                sums[k] += t-mark
                mark = t
            for i in range(n):
                with torch.no_grad():
                    a = int(m.distribution(tensor_observation(obs,m.device)).sample()[0])
                lap('action')
                nxt,r,done,dt = env.step(a)
                lap('env')
                replay.extend(transition(obs,a,nxt,r,done,dt))
                obs = nxt
                lap('insert')
                if (i+1) % d['update_every'] == 0:
                    batch = replay.sample().to(m.device)
                    lap('sample_and_transfer')
                    m.update(batch)
                    lap('update')
            sync()
            return dict(n=n, seconds=clock()-start, parts=dict(sums))
        block(64, False)
        out['natural'] = [block(160,False) for _ in range(3)]
        out['fenced'] = [block(128,True) for _ in range(2)]
        b = replay.sample().to(m.device)
        out['batch_bytes'] = sum(v.numel()*v.element_size() for v in b.values(True,True))
        parts = defaultdict(float)
        def meter(name, fn):
            def wrapped(*args, **kwargs):
                sync()
                t = clock()
                val = fn(*args, **kwargs)
                sync()
                parts[name] += clock()-t
                return val
            return wrapped
        with ExitStack() as stack:
            for obj,name,label in [(m.loss,'forward','loss_forward'),(torch.Tensor,'backward','backward'),
                    (torch.nn.utils,'clip_grad_norm_','clip'),(m.optimizer,'step','adamw'),
                    (m.updater,'step','soft_target')]:
                stack.enter_context(patch.object(obj,name,meter(label,getattr(obj,name))))
            sync()
            t = clock()
            for _ in range(16): m.update(b)
            sync()
            out['update_16'] = dict(total=clock()-t, parts=dict(parts))
        td = tensor_observation(obs,m.device)
        Path(dest).write_text(json.dumps(out,indent=2))
        # The actual stochastic policy, with binding/mode reused only between updates.
        def sample4(fast):
            with torch.no_grad(), (module_mode(m.actor,False) if fast and m.explicit_modes else nullcontext()), (m.loss.actor_network_params.to_module(m.actor) if fast else nullcontext()):
                draws=[]
                for _ in range(4):
                    dist = m.actor.get_dist(td) if fast else m.distribution(td)
                    draws.append(int(dist.sample()[0]))
                return draws
        series = {}
        for fast in [False,True,True,False]:
            torch.manual_seed(17)
            sync()
            t = clock()
            draws = sum((sample4(fast) for _ in range(48)),[])
            sync()
            series.setdefault(str(fast),[]).append(clock()-t)
            if fast and draws != reference: raise ValueError('stochastic action parity')
            if not fast: reference = draws
        out['binding_192_actions'] = series
        out['binding_matched_draws'] = len(reference)
        Path(dest).write_text(json.dumps(out,indent=2))
        # Operator attribution is collected separately to avoid timing profiler overhead.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(2): m.update(b)
            sync()
        avg = prof.key_averages()
        top = lambda key: [[e.key,e.count,round(getattr(e,key),1)] for e in
                           sorted(avg,key=lambda e:getattr(e,key),reverse=True)[:15]]
        out['ops_2_updates'] = {k:top(k) for k in ['self_cpu_time_total','self_device_time_total']}
        Path(dest).write_text(json.dumps(out,indent=2))
        # Compare both attention dispatches from identical weights, optimizer, batch and RNG.
        import copy
        weights=copy.deepcopy(m.loss.state_dict())
        opt=copy.deepcopy(m.optimizer.state_dict())
        dispatch=[]
        original=torch.backends.mha.get_fastpath_enabled()
        try:
            for mode in (True,False,False,True):
                m.loss.load_state_dict(weights)
                m.optimizer.load_state_dict(copy.deepcopy(opt))
                torch.manual_seed(37)
                torch.backends.mha.set_fastpath_enabled(mode)
                for _ in range(4): m.update(b)
                sync()
                t=clock()
                for _ in range(16): losses=m.update(b)
                sync()
                record=dict(mode=mode,seconds=clock()-t,losses=losses)
                with torch.no_grad():
                    logits=m.surfaces(b['features'])
                    if not dispatch:
                        ref=logits.clone()
                        refweights=copy.deepcopy(m.loss.state_dict())
                    record['logit_delta']=float((logits-ref).abs().max())
                    record['weight_delta']=max(float((v-refweights[k]).abs().max())
                        for k,v in m.loss.state_dict().items() if v.is_floating_point() and v.numel())
                dispatch.append(record)
        finally:
            torch.backends.mha.set_fastpath_enabled(original)
        out['dispatch_abba']=dispatch
        Path(dest).write_text(json.dumps(out,indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
