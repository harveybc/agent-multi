"""Project a donor-expanded actor into an ordinary selected-input architecture."""
import re
import torch


def transfer_expanded_actor(expanded, destination, universe, selected, group_order):
    names=universe['list']['columns']; groups=universe['representation']['resolved']
    columns={g:[names[j] for j in ix] for g,ix in zip(groups['names'],groups['indices'])}
    dg=selected['representation']['resolved']; active=selected['list']['columns']
    source=expanded.state_dict(); target=destination.state_dict(); copied=[]
    width=expanded.encoder.fusion.conv.in_channels//len(group_order)
    if width*len(group_order)!=expanded.encoder.fusion.conv.in_channels:
        raise ValueError('EXPORT_FUSION_WIDTH')
    for g in dg['names']:
        if g not in group_order:raise ValueError('EXPORT_GROUP_MISSING')
        branch=expanded.encoder.branches[group_order.index(g)]
        if not isinstance(branch,torch.nn.Sequential):raise ValueError('EXPORT_CENTERED_GROUP')
    # Entirely removed original groups can have nonzero constant responses.
    for i,g in enumerate(group_order):
        if g not in dg['names']:
            b=expanded.encoder.branches[i];p=next(b.parameters())
            with torch.no_grad():v=b(p.new_zeros(1,len(columns[g]),expanded.encoder.shape[0]))
            if torch.count_nonzero(v):raise ValueError('EXPORT_NONZERO_REMOVED_GROUP')
    with torch.no_grad():
        for key,value in target.items():
            if key.startswith('encoder.indices_'):continue
            origin=key;m=re.match(r'encoder.branches.(\d+)\.(.*)',key)
            if m:
                i=int(m[1]);g=dg['names'][i];si=group_order.index(g)
                origin=f'encoder.branches.{si}.{m[2]}'
            old=source[origin]
            if m and m[2]=='0.conv.weight':
                keep=[columns[g].index(active[j]) for j in dg['indices'][i]]
                old=old[:,keep,:]
            elif key=='encoder.fusion.conv.weight':
                keep=[group_order.index(g)*width+j for g in dg['names'] for j in range(width)]
                old=old[:,keep,:]
            if old.shape!=value.shape:raise ValueError('EXPORT_SHAPE:'+key)
            value.copy_(old)
            if not torch.equal(value,old):raise ValueError('EXPORT_VALUE:'+key)
            copied.append(key)
    return dict(copied_exact=len(copied),names=copied,input_features=len(active),
                groups=len(dg['names']),physical_input_width=destination.encoder.shape[-1])
