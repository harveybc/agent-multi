"""Recipe-pinned modular genome; inactive temporal genes have explicit receipts."""
import copy

from app.architecture_gene_contract import decode, encode, value_at
from app.frozen_recipe_inputs import validate_recipe
from agent_plugins.temporal_stage_config import stage_config


def make_contract(v1, base, recipe):
    validate_recipe(recipe)
    config=copy.deepcopy(base)
    architecture=config['architecture']
    names=recipe['representation']['resolved']['names']
    if any(k in architecture for k in ('group_configs','core_config')):
        raise ValueError('MODULAR_BASE_MUST_BE_SHARED_352')
    architecture['actor_head_width']=architecture.get('actor_head_width',architecture['head_width'])
    for key,value in dict(head_depth=1,actor_head_depth=1,head_activation='silu',actor_head_activation='silu').items():
        architecture.setdefault(key,value)
    architecture['group_configs']=[dict(width=architecture['branch_width'],blocks=architecture['branch_blocks'],
        kind='tcn' if k=='conv' else 'pointwise',kernel=3 if k=='conv' else 1,
        dilation_base=2,activation='silu') for k in recipe['representation']['resolved']['kinds']]
    architecture['core_config']=dict(width=architecture['core_width'],blocks=architecture['core_blocks'],
        kind='tcn',kernel=3,dilation_base=2,activation='silu')
    shadowed={'architecture.'+k for k in ('branch_width','branch_blocks','core_width','core_blocks')}
    genes=[copy.deepcopy(g) for g in v1['genes'] if g['path'] not in shadowed]
    groups=[('architecture.group_configs.'+str(i),False) for i in range(len(names))]
    groups.append(('architecture.core_config',True))
    for path,core in groups:
        domains=dict(width=[8,16,32,64] if core else [4,8,16,32],
            blocks=[1,2,3,4] if core else [1,2,3],kind=['tcn','pointwise'],kernel=[1,3,5],
            dilation_base=[1,2,3],activation=['silu','relu','gelu','tanh'])
        genes.extend(dict(path=path+'.'+key,kind='choice',values=values) for key,values in domains.items())
    for prefix in ('architecture.head_','architecture.actor_head_'):
        genes.extend([dict(path=prefix+'depth',kind='choice',values=[1,2,3]),
                      dict(path=prefix+'activation',kind='choice',values=['silu','relu','gelu','tanh'])])
    space=dict(schema='architecture_gene_contract.v2',status='CONSUMPTION_PREFLIGHT_NOT_SEARCH',
        recipe_identity=recipe['identity'],group_order=list(names),genes=genes,
        shadowed_shared_paths=sorted(shadowed),fixed=copy.deepcopy(v1['fixed']),
        unsupported=['independent encoder topologies per actor/critic','normalization alternatives',
                     'arbitrary non-geometric dilation genes','training-loop genes','strategy changes'])
    return space,config


def decode_modular(space, raw, base, recipe):
    validate_recipe(recipe)
    if (space.get('schema')!='architecture_gene_contract.v2' or space['recipe_identity']!=recipe['identity']
            or space['group_order']!=recipe['representation']['resolved']['names']):
        raise ValueError('MODULAR_RECIPE_IDENTITY')
    requested=decode(space,raw,base)
    effective=copy.deepcopy(requested)
    active=[True]*len(space['genes']); inactive=[]
    paths={g['path']:i for i,g in enumerate(space['genes'])}
    stages=[('architecture.group_configs.'+str(i),s) for i,s in enumerate(effective['architecture']['group_configs'])]
    stages.append(('architecture.core_config',effective['architecture']['core_config']))
    for path,s in stages:
        if s['kind']=='pointwise':
            active[paths[path+'.kernel']]=False
            inactive.append(dict(path=path+'.kernel',requested=s['kernel'],effective=1,reason='pointwise'))
            s['kernel']=1
        if s['kernel']==1 or s['blocks']==1:
            active[paths[path+'.dilation_base']]=False
            frozen=value_at(base,path+'.dilation_base')
            inactive.append(dict(path=path+'.dilation_base',requested=s['dilation_base'],effective=frozen,
                                 reason='kernel1_or_single_block'))
            # Preserve the baseline dispatch even where dilation is algebraically inert.
            s['dilation_base']=frozen
        stage_config(s,width=s['width'],blocks=s['blocks'])
    return dict(raw_genome=list(raw),requested=requested,effective=effective,active_gene_mask=active,
                inactive_genes=inactive)


def baseline_genome(space, config):
    return encode(space,config)
