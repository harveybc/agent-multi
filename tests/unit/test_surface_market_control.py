import torch

from tools.run_surface_market_control import fit_constant_embeddings


def test_constant_market_is_fit_only_and_bank_balanced():
    z=torch.tensor([[[1.],[3.],[10.],[999.]],[[2.],[6.],[20.],[888.]]])
    records=[{'role':'fit'}]*3+[{'role':'monitor'}]
    banks=[[0,1],[2],[3]]
    constant,mean=fit_constant_embeddings(z,banks,records)
    torch.testing.assert_close(mean,torch.tensor([[[6.]],[[12.]]]))
    assert torch.equal(constant,mean.expand_as(constant))
    changed=z.clone();changed[:,3]=0
    other,_=fit_constant_embeddings(changed,banks,records)
    assert torch.equal(constant,other)
