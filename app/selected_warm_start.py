"""Selected-channel SAC transfer with unchanged surviving group topology."""
import re
from pathlib import Path
import torch
from app.rl_subset import subset_recipe
from app.frozen_recipe_inputs import file_sha
from agent_plugins.sac_critic_regularization import module_mode


def transfer_selected(source, dest):
    selected=dest.recipe['list']['columns']
    if dest.recipe!=subset_recipe(source.recipe,selected):
        raise ValueError('TRANSFER_RECIPE')
    sg=source.recipe['representation']['resolved'];dg=dest.recipe['representation']['resolved']
    if sg['names']!=dg['names'] or sg['kinds']!=dg['kinds']:
        raise ValueError('TRANSFER_GROUP_TOPOLOGY_CHANGED')
    if source.effective['architecture']!=dest.effective['architecture']:
        raise ValueError('TRANSFER_ARCHITECTURE')
    names=source.recipe['list']['columns'];keep=set(selected)
    positions={i:[j for j,k in enumerate(indices) if names[k] in keep]
               for i,indices in enumerate(sg['indices'])}
    old=source.loss.state_dict();new=dest.loss.state_dict()
    if old.keys()!=new.keys(): raise ValueError('TRANSFER_STATE_KEYS')
    sliced=[];copied=[];buffers=[]
    with torch.no_grad():
        for name,target in new.items():
            if re.search(r'\.encoder\.indices_\d+$',name):
                buffers.append(name);continue
            value=old[name]
            match=re.search(r'\.encoder\.branches\.(\d+)\.0\.conv\.weight$',name)
            if match:
                i=int(match[1]);ix=positions[i]
                if len(ix)!=len(sg['indices'][i]):
                    value=value.index_select(-2,torch.tensor(ix,device=value.device))
                    sliced.append(name)
            if value.shape!=target.shape: raise ValueError('TRANSFER_SHAPE:'+name)
            target.copy_(value)
            if not torch.equal(target,value): raise ValueError('TRANSFER_VALUES:'+name)
            copied.append(name)
    return dict(retained_tensors_exact=True,sliced_tensors=len(sliced),sliced_names=sliced,
                copied_tensors=len(copied),remapped_index_buffers=buffers,
                source_features=len(names),selected_features=len(selected),groups=len(sg['names']))


def configure_head_only(model):
    trained=[];frozen=[]
    for name,param in model.loss.named_parameters():
        train='.head.' in name and not name.startswith('target_')
        param.requires_grad_(train)
        (trained if train else frozen).append(name)
    if not trained or not frozen: raise ValueError('TRANSFER_TRAINABLE_PARTITION')
    model._warm_frozen={n:p.detach().clone() for n,p in model.loss.named_parameters() if n in frozen}
    # Keep optimizer parameter-group layout for ordinary bundle/recovery reload.
    # AdamW skips tensors with no gradient, including their weight decay.
    return dict(trainable=trained,frozen=frozen,temperature='copied_and_frozen',
                target_networks='copied_then_standard_soft_update',optimizer='fresh_AdamW')


def verify_frozen(model):
    params=dict(model.loss.named_parameters())
    for name,expected in model._warm_frozen.items():
        if not torch.equal(params[name],expected):raise ValueError('FROZEN_WEIGHT_CHANGED:'+name)


def target_surfaces(model,features):
    values=[]
    with module_mode(model.loss.qvalue_network,False):
        for i in range(2):
            with model.loss.target_qvalue_network_params[i].to_module(model.loss.qvalue_network):
                values.append(model.q_net.surface(features))
    return torch.stack(values)


def export_components(model,root):
    from tools.run_discrete_curve_probe import write
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    components={}
    for name,value in model.loss.state_dict().items():
        role=name.split('.')[0]
        m=re.search(r'\.encoder\.branches\.(\d+)\.',name)
        component=('extractor_'+m[1] if m else 'fusion' if '.encoder.fusion.' in name else
                   'attention' if '.encoder.post_fusion.' in name else
                   'core' if '.encoder.core.' in name else 'head' if '.head.' in name else 'state')
        components.setdefault(role+'/'+component,{})[name]=value.detach().cpu().clone()
    files={}
    for name,state in components.items():
        path=root/(name+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(state,path);files[str(path.relative_to(root))]=file_sha(path)
    write(root/'MANIFEST.json',dict(recipe=model.recipe,effective=model.effective,files=files,
          scope='component tensors from one coherent SAC checkpoint; full bundle is authoritative'))
    return files
