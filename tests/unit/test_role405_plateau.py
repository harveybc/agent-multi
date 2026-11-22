import copy
import unittest
import torch
from tools.role405_plateau import POLICY,observe,restore,snapshot


class PlateauTest(unittest.TestCase):
    def test_schedule(self):
        lr=POLICY['initial_lr'];bad=0;events=[]
        for epoch in range(1,181):
            bad,rate=observe(epoch,epoch==1,bad,lr,False)
            if rate is not None:events.append((epoch,rate));lr=rate
        self.assertEqual(events,[(40,6e-5),(60,1.2e-5),(80,2.4e-6),(100,4.8e-7)])
        self.assertEqual(observe(58,True,18,.0003,False),(0,None))
        self.assertIsNone(observe(179,False,19,.0003,True)[1])

    def test_optimizer_and_resume(self):
        torch.manual_seed(11)
        model=torch.nn.Linear(2,1)
        opt=torch.optim.AdamW(model.parameters(),lr=.0003)
        def step():
            opt.zero_grad();model(torch.ones(1,2)).square().sum().backward();opt.step()
        step()
        weights=copy.deepcopy(model.state_dict());moments=snapshot(opt)
        step();restore(model,opt,weights,moments,.00006)
        self.assertEqual(opt.param_groups[0]['lr'],.00006)
        restart=copy.deepcopy((model.state_dict(),opt.state_dict()))
        step();expected=copy.deepcopy(model.state_dict())
        model.load_state_dict(restart[0]);opt.load_state_dict(restart[1]);step()
        for k,v in expected.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        for slot in moments['state'].values():self.assertEqual(int(slot['step']),1)
