import numpy as np
import pytest
import torch

from app.opening_boundary_refinement import choose_boundary_panel, margin_derivatives, pressure_summary


def trace_rows():
    rows=[]
    start=1577836800000000000
    for i in range(100):
        winner=i%3
        z=[0.,-.2,-.3];z[winner]=.01*(i+1)
        rows.append(dict(timestamp=start+i*24*3600*10**9,year=2020,row=i,
                         position=0,allowed=True,logits=z))
    rows.extend([dict(rows[i],row=100+i,timestamp=start+(100+i)*24*3600*10**9,
                      position=1 if i%2 else -1,allowed=False) for i in range(10)])
    rows.extend([dict(rows[i],row=110+i,timestamp=start+(110+i)*24*3600*10**9,
                      allowed=False) for i in range(5)])
    return rows


def test_selection_is_deterministic_and_ignores_outcomes():
    rows=trace_rows();a=choose_boundary_panel(rows)
    b=choose_boundary_panel([dict(r,future_profit=-99999) for r in reversed(rows)])
    assert [r['timestamp'] for r in a]==[r['timestamp'] for r in b]
    assert len(a)==len({r['timestamp'] for r in a}) and len(a)<=38
    for d in (1,2):
        for winner in (0,d):
            key=f'boundary_{d}_winner_{winner}'
            selected=[r for r in a if key in r['boundary_strata']]
            assert len(selected)==6
            assert all(np.argmax(r['logits'])==winner and r['allowed'] for r in selected)
            ts=sorted(r['timestamp'] for r in selected)
            assert min(np.diff(ts))>=24*3600*10**9


@pytest.mark.parametrize('field,value',[('year',2024),('position',2),('logits',[0.,float('nan'),0.]),('allowed',1)])
def test_bad_trace_refused(field,value):
    rows=trace_rows();rows[0][field]=value
    with pytest.raises(ValueError):choose_boundary_panel(rows)


def test_gradient_sign_matches_small_sgd_step_and_components():
    torch.manual_seed(7)
    layer=torch.nn.Linear(2,3,dtype=torch.float64)
    x=torch.tensor([[.2,1.],[1.,-.3]],dtype=torch.float64)
    z=layer(x);q=torch.tensor([[0.,1.,-1.],[1.,0.,2.]],dtype=torch.float64)
    lp=z.log_softmax(-1);value=-(lp.exp()*q).sum(-1).mean();entropy=.1*(lp.exp()*lp).sum(-1).mean()
    params=list(layer.parameters());g=torch.autograd.grad(value+entropy,params,retain_graph=True)
    report=margin_derivatives(z,params,value,entropy)
    assert np.allclose(np.array(report['value'])+report['entropy'],report['total'])
    before=(z[:,1:]-z[:,:1]).detach().numpy()
    with torch.no_grad():
        for p,grad in zip(params,g):p.add_(grad,alpha=-1e-5)
    after=(layer(x)[:,1:]-layer(x)[:,:1]).detach().numpy()
    assert np.allclose((after-before)/1e-5,report['total'],atol=1e-8)


def test_pressure_report_separates_negative_advantage_from_model_q():
    rows=[dict(row=1,role='fit',position=0,allowed=True)]
    g=dict(rows=[1],q=[[0,2,-1]],native_reward=[[0,-1,1]],native_cash=[[0,-2,3]],
           value=[[2,-2]],entropy=[[0,0]],total=[[2,-2]])
    actual=dict(rows=[1],margin_delta=[[-.1,.1]])
    out=pressure_summary(rows,g,actual)['fit:flat_allowed']
    assert out['q_favors_harmful']==1
    assert out['pressure']['total']['favors_harmful']==1
    assert out['pressure']['total']['suppresses_helpful']==1
    assert out['pressure']['adam_margin']['favors_harmful']==0
    with pytest.raises(ValueError):pressure_summary(rows,g,dict(actual,rows=[2]))
