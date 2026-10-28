"""Opt-in critic-only ablations over native SB3 SAC policy/optimizers."""
from copy import deepcopy
import math

import torch.nn as nn
from stable_baselines3.sac.policies import SACPolicy


PROFILES = {
    'dropout005': dict(dropout=.05, l2=0., batch_norm=False),
    'l2_1e6': dict(dropout=0., l2=1e-6, batch_norm=False),
    'dropout005_l2_1e6': dict(dropout=.05, l2=1e-6, batch_norm=False),
    'batchnorm': dict(dropout=0., l2=0., batch_norm=True),
}


def validate_profile(value):
    if (not isinstance(value,dict) or set(value)!={'dropout','l2','batch_norm'}
            or type(value['batch_norm']) is not bool
            or any(type(value[k]) not in (int,float) or not math.isfinite(value[k])
                   for k in ('dropout','l2'))
            or value not in PROFILES.values()):
        raise ValueError('SAC_REGULARIZATION_APPROVED_PROFILE_REQUIRED')
    return deepcopy(value)


class RegularizedSACPolicy(SACPolicy):
    def __init__(self,*args,regularization,**kwargs):
        self.regularization=validate_profile(regularization)
        super().__init__(*args,**kwargs)
        for group in self.critic.optimizer.param_groups:
            group['weight_decay']=self.regularization['l2']

    def make_critic(self,features_extractor=None):
        critic=super().make_critic(features_extractor)
        for index,network in enumerate(critic.q_networks):
            layers=[]
            for i,layer in enumerate(network):
                layers.append(layer)
                if isinstance(layer,nn.Linear) and i<len(network)-1 and self.regularization['batch_norm']:
                    layers.append(nn.BatchNorm1d(layer.out_features))
                if isinstance(layer,nn.ReLU) and self.regularization['dropout']:
                    layers.append(nn.Dropout(self.regularization['dropout']))
            network=nn.Sequential(*layers)
            setattr(critic,f'qf{index}',network)
            critic.q_networks[index]=network
        return critic

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),regularization=self.regularization)


def gpu_smoke(root):
    """Finite synthetic admission after the previous GPU owner has exited."""
    import json
    import os
    from pathlib import Path
    import tempfile
    import numpy as np
    import gymnasium
    import torch
    from stable_baselines3 import SAC
    from agent_plugins.sac_agent import Plugin, _policy_tensor_hash
    from tools.curated_rl_campaign import digest
    from tools.verify_rl_static_engineering import save_status
    root=Path(root)
    contract=root/'CONTRACT.json'
    body=json.loads(contract.read_text())
    profile=validate_profile(body['common_config']['sac_regularization'])
    uuid=body['execution']['gpu_evidence']['assigned_gpu_uuid']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid or not torch.cuda.is_available():
        return 1
    path=root/'GPU_REGULARIZATION_SMOKE.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    identity=dict(contract_sha256=digest(contract),profile=profile,gpu_uuid=uuid)
    if previous and previous.get('identity')!=identity:
        return 78
    if previous.get('state')=='VERIFIED':
        return 0
    attempt=previous.get('attempt',0)+1
    if attempt>2:
        return 78
    owner=gymnasium.make('Pendulum-v1')
    try:
        model=Plugin().build(owner,dict(device='cuda',net_arch=[32,32],train_seed=7,
            sac_regularization=profile,buffer_size=64,learning_starts=4,batch_size=8))
        model.learn(24)
        if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
            raise ValueError('REGULARIZATION_NONFINITE_SMOKE')
        obs=np.zeros((1,3),dtype=np.float32)
        action=model.predict(obs,deterministic=True)[0]
        before=_policy_tensor_hash(model.policy)
        with tempfile.TemporaryDirectory(prefix='regularization-gpu-smoke-') as temp:
            checkpoint=Path(temp)/'policy.zip';model.save(checkpoint)
            restored=SAC.load(checkpoint,device='cuda')
            if (_policy_tensor_hash(restored.policy)!=before
                    or not np.array_equal(action,restored.predict(obs,deterministic=True)[0])):
                raise ValueError('REGULARIZATION_GPU_ROUNDTRIP_FAILED')
        save_status(path,dict(schema='sac_regularization_gpu_smoke.v1',
            classification='NON_GOVERNING_SYNTHETIC_GPU',state='VERIFIED',identity=identity,
            attempt=attempt,transitions=24,gradient_updates=model._n_updates,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_version=torch.__version__))
        return 0
    except (ValueError,RuntimeError) as exc:
        save_status(path,dict(schema='sac_regularization_gpu_smoke.v1',
            classification='NON_GOVERNING_SYNTHETIC_GPU',state='FAILED',identity=identity,
            attempt=attempt,error_type=type(exc).__name__))
        return 1 if attempt<2 else 78
    finally:
        owner.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-root',required=True)
    raise SystemExit(gpu_smoke(parser.parse_args().smoke_root))
