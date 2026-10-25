"""Actor/critic encoder isolation with identical copied initial representations."""
import copy

from agent_plugins.qualified_orders import QualifiedOrderPolicy
from agent_plugins.hybrid_orders import OrderDistribution


class IndependentOrderPolicy(QualifiedOrderPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Copy, not another random initialization: the initial functions match.
        self.critic_encoder=copy.deepcopy(self.features_extractor)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def set_training_mode(self, mode):
        super().set_training_mode(mode)
        self.critic_encoder.eval()

    def _parts(self, obs):
        pi=self.mlp_extractor.forward_actor(self.features_extractor(obs))
        vf=self.mlp_extractor.forward_critic(self.critic_encoder(obs))
        mask=obs['action_mask'].bool()
        dist=OrderDistribution(self.commands(pi),self.risks(pi).reshape(-1,2,6),mask,self.learned_risk)
        route=mask[:,3].long()+2*mask[:,4].long()
        return dist,self.values(vf).gather(1,route[:,None])
