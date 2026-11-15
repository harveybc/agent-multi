"""CPU-only modular constructor/update witnesses; never a trading fitness run."""
import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from tensordict import TensorDict

from app.frozen_recipe_inputs import file_sha
from app.structural_refit import build_model, load_bundle
from app.modular_gene_contract import make_contract, decode_modular, baseline_genome
from app.architecture_gene_contract import value_at
from app.rl_subset import digest
from tools.audit_architecture_gene_contract import observe


def configurations(base, recipe):
    yield 'baseline', copy.deepcopy(base)
    for i, name in enumerate(recipe['representation']['resolved']['names']):
        c=copy.deepcopy(base)
        c['architecture']['group_configs']=[None]*len(recipe['representation']['resolved']['names'])
        c['architecture']['group_configs'][i]=dict(width=4,blocks=1,kind='tcn',kernel=5,dilations=[2],activation='relu')
        yield 'group_'+str(i),c
    for kind in ('tcn','pointwise'):
        c=copy.deepcopy(base)
        c['architecture']['core_config']=dict(width=24,blocks=3,kind=kind,
            kernel=3 if kind=='tcn' else 1,dilations=[1,3,5],activation='gelu')
        yield 'core_'+kind,c
    c=copy.deepcopy(base)
    c['architecture'].update(head_width=12,head_depth=3,head_activation='relu',
        actor_head_width=4,actor_head_depth=2,actor_head_activation='tanh')
    yield 'separate_heads',c


def assert_modules(model, config, space):
    actual=observe(model)
    encoder=model.actor_net.encoder
    stages=[('architecture.group_configs.'+str(i),branch[1:],branch[0].conv.out_channels)
            for i,branch in enumerate(encoder.branches)]
    stages.append(('architecture.core_config',encoder.core,encoder.fusion.conv.out_channels))
    for path,blocks,width in stages:
        expected=value_at(config,path)
        actual.update({path+'.width':width,path+'.blocks':len(blocks)})
        for i,block in enumerate(blocks):
            for conv in (block.first.conv,block.second.conv):
                if (conv.kernel_size!=(expected['kernel'],) or
                    conv.dilation!=(expected['dilation_base']**i,)):
                    raise ValueError('GENE_TEMPORAL_NOT_CONSUMED:'+path)
        actual[path+'.kernel']=blocks[0].first.conv.kernel_size[0]
        actual[path+'.activation']=type(blocks[0].activation).__name__.lower()
        # Mode selects a kernel constraint, not a distinct module class.
        if expected['kind']=='pointwise' and actual[path+'.kernel']!=1:
            raise ValueError('GENE_POINTWISE_NOT_CONSUMED')
    for prefix,head in [('architecture.head_',model.q_net.head),('architecture.actor_head_',model.actor_net.head)]:
        actual[prefix+'depth']=(len(head)-1)//2
        actual[prefix+'activation']=type(head[1]).__name__.lower()
    for gene in space['genes']:
        path=gene['path']
        if path.endswith(('.kind','.dilation_base')):continue  # checked through both convolutions above
        expected=value_at(config,path);values=actual[path]
        if not isinstance(values,list):values=[values]
        for value in values:
            if isinstance(expected,str):ok=value==expected
            else:ok=math.isclose(value,expected,rel_tol=1e-6,abs_tol=1e-8)
            if not ok:raise ValueError('GENE_MODULE_NOT_CONSUMED:'+path)
    if encoder.fusion.conv.in_channels!=sum(b[0].conv.out_channels for b in encoder.branches):
        raise ValueError('GENE_FUSION_NOT_CONSUMED')


def audit(bundle, output):
    if output.exists():raise ValueError('NEW_AUDIT_ROOT_REQUIRED')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    original,manifest=load_bundle(bundle,'cpu')
    with np.load(bundle/'PROBE.npz') as probe:
        x=torch.tensor(probe['inputs']); saved=probe['logits']
    with torch.inference_mode(): actual=original.surfaces(x).numpy()
    base=json.loads((bundle/'MODEL_CONFIG.json').read_text())
    recipe=original.recipe
    # Compare old/new code on the SAME host. Exported CPU probes can differ by
    # backend dispatch; record that separately, never call it byte-identical.
    reference_revision='4531474c99685afaaa0775c179bf772c5b1d04dc'
    modules={}
    try:
        for name in ('double_tcn_probe','three_signal_sac'):
            full='agent_plugins.'+name
            modules[full]=sys.modules[full]
            mod=types.ModuleType(full)
            mod.__file__='agent_plugins/'+name+'.py'
            source=subprocess.check_output(['git','show',reference_revision+':'+mod.__file__],text=True)
            sys.modules[full]=mod
            exec(compile(source,mod.__file__,'exec'),mod.__dict__)
        legacy=sys.modules['agent_plugins.three_signal_sac'].ThreeSignalSAC(recipe,
            architecture=base['architecture'],lr=base['learning_rate'],gamma=base['gamma'],
            alpha=base['alpha'],entropy_fraction=base['entropy_fraction'],tau=base['tau'],
            weight_decay=base['weight_decay'],critic_regularization=base.get('critic_regularization'))
        legacy.load(bundle/'model.pt')
        with torch.inference_mode(): reference=legacy.surfaces(x).numpy()
    finally:
        sys.modules.update(modules)
    if not np.array_equal(actual,reference):raise ValueError('SAME_HOST_BASELINE_PARITY')
    if not np.array_equal(actual.argmax(-1),saved.argmax(-1)):raise ValueError('EXPORTED_PROBE_ACTIONS')
    output.mkdir(parents=True)
    v1=json.loads(Path('examples/config/phase2_architecture_gene_contract.json').read_text())
    space,explicit=make_contract(v1,base,recipe)
    raw=baseline_genome(space,explicit)
    decoded=decode_modular(space,raw,explicit,recipe)
    shared=build_model(recipe,base,seed=7,device='cpu')
    normalized=build_model(recipe,decoded['effective'],seed=7,device='cpu')
    with torch.inference_mode():
        if not torch.equal(shared.surfaces(x),normalized.surfaces(x)):
            raise ValueError('MODULAR_BASELINE_DECODE')
    requests=[];constructed=set()
    for i,g in enumerate(space['genes']):
        points=[0.,.5,1.]
        if g['kind']=='choice':points+=[(j+.5)/len(g['values']) for j in range(len(g['values']))]
        for value in points:
            candidate=list(raw);candidate[i]=value
            request=decode_modular(space,candidate,explicit,recipe)
            key=digest(request['effective'])
            if key not in constructed:
                witness=build_model(recipe,request['effective'],seed=7,device='cpu')
                assert_modules(witness,request['effective'],space)
                with torch.inference_mode():
                    if not torch.isfinite(witness.surfaces(x[:2])).all():raise ValueError('GENE_NONFINITE_LOGITS')
                constructed.add(key)
                del witness
            requests.append(dict(path=g['path'],normalized=value,active=request['active_gene_mask'][i],
                                 inactive_genes=request['inactive_genes'],phenotype_id=key))
        print('gene',i+1,'/',len(space['genes']),'verified',flush=True)
    for name,value in [('GENE_CONTRACT_V2.json',space),('EXPLICIT_BASE.json',explicit),('DECODE_WITNESSES.json',requests)]:
        (output/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    results=[]
    for name,config in configurations(base,recipe):
        m=build_model(recipe,config,seed=7,device='cpu')
        e=m.actor_net.encoder
        z=x[:2]; changed=z.clone();changed[:,7:]+=1
        e.eval()
        with torch.no_grad():
            if not torch.equal(e.sequence({'features':z})[:,:,:7],e.sequence({'features':changed})[:,:,:7]):
                raise ValueError('MODULAR_PREFIX:'+name)
        # Fixture transitions are only a learning-path witness, never a profit measurement.
        obs=z.repeat(16,1,1)
        batch=TensorDict(dict(features=obs,mask=torch.ones(32,3,dtype=torch.bool),action=torch.arange(32)%3),[32])
        batch['next']=TensorDict(dict(features=obs*.9,mask=torch.ones(32,3,dtype=torch.bool),
            reward=torch.linspace(-.1,.1,32)[:,None],done=torch.zeros(32,1,dtype=torch.bool),
            terminated=torch.zeros(32,1,dtype=torch.bool)),[32])
        before={k:v.detach().clone() for k,v in m.loss.named_parameters()}
        losses=m.update(batch)
        if not all(np.isfinite(list(losses.values()))):raise ValueError('MODULAR_NONFINITE')
        changed_names=[k for k,v in m.loss.named_parameters() if not torch.equal(v,before[k])]
        if not any(k.startswith('actor_network_params') for k in changed_names):raise ValueError('ACTOR_UNCHANGED')
        if not any(k.startswith('qvalue_network_params') for k in changed_names):raise ValueError('CRITIC_UNCHANGED')
        with torch.inference_mode():a=m.surfaces(z).clone();q=m.q_surfaces(z).clone()
        path=output/(name+'.pt');m.save(path)
        cold=build_model(recipe,config,seed=99,device='cpu');cold.load(path)
        with torch.inference_mode():
            if not torch.equal(a,cold.surfaces(z)) or not torch.equal(q,cold.q_surfaces(z)):
                raise ValueError('MODULAR_COLD_RELOAD')
        results.append(dict(name=name,requested=config,resolved=e.resolved_architecture,
            parameter_counts={n:sum(p.numel() for p in module.parameters()) for n,module in
                [('actor',m.actor_net),('critic_template',m.q_net)]},
            updated_parameters=changed_names,losses=losses,optimizer_steps=1,batch_size=32,
            checkpoint_sha256=file_sha(path),prefix_exact=True,cold_reload_exact=True))
        print(name,'PASS',flush=True)
    result=dict(schema='modular_architecture356.v1',same_host_baseline_logits_exact=True,
        reference_revision=reference_revision,exported_probe_logits_exact=np.array_equal(actual,saved),
        exported_probe_max_abs_error=float(np.max(abs(actual-saved))),exported_probe_actions_exact=True,
        bundle_manifest_sha256=file_sha(bundle/'MANIFEST.json'),recipe_identity=manifest['recipe_identity'],
        features=recipe['list']['n'],groups=len(recipe['representation']['resolved']['names']),
        gene_contract=dict(genes=len(space['genes']),decode_requests=len(requests),
            baseline_logits_exact=True,constructed_phenotypes=len(constructed),all_decoded_and_constructed=True,
            sha256=file_sha(output/'GENE_CONTRACT_V2.json'),witness_sha256=file_sha(output/'DECODE_WITNESSES.json')),
        source_files={str(p):file_sha(p) for p in map(Path,[
            'agent_plugins/temporal_stage_config.py','agent_plugins/double_tcn_probe.py',
            'agent_plugins/three_signal_sac.py','tools/audit_modular_architecture.py',
            'app/modular_gene_contract.py','app/architecture_gene_contract.py'])},
        cases=results,device='cpu',training_epochs=0,trading_fitness=None,deap_launched=False)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();audit(a.bundle,a.output)
