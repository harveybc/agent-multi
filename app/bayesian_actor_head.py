"""Bounded variational last-layer intervention with explicit inference draws."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LastLayer(nn.Module):
    def __init__(self, linear, bayesian=False, prior_std=0.1, posterior_std=0.001):
        super().__init__()
        if not isinstance(linear, nn.Linear) or linear.bias is None:
            raise ValueError('BAYES_LINEAR_REQUIRED')
        if not 0 < posterior_std <= prior_std or not math.isfinite(prior_std):
            raise ValueError('BAYES_SCALE')
        self.bayesian = bayesian
        self.register_buffer('prior_weight', linear.weight.detach().clone())
        self.register_buffer('prior_bias', linear.bias.detach().clone())
        self.register_buffer('prior_std', linear.weight.new_tensor(prior_std))
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone())
        rho = math.log(math.expm1(posterior_std))
        if bayesian:
            self.weight_rho = nn.Parameter(torch.full_like(self.weight, rho))
            self.bias_rho = nn.Parameter(torch.full_like(self.bias, rho))

    def mean_logits(self, x):
        return F.linear(x, self.weight, self.bias)

    def draws(self, count, seed):
        if count < 2 or count % 2:
            raise ValueError('BAYES_EVEN_DRAWS')
        generator = torch.Generator(device=self.weight.device).manual_seed(seed)
        result = []
        for p in (self.weight, self.bias):
            e = torch.randn((count // 2, *p.shape), device=p.device, dtype=p.dtype,
                            generator=generator)
            result.append(torch.cat((e, -e)))
        return tuple(result)

    def logits(self, x, noise=None):
        if not self.bayesian:
            return self.mean_logits(x).unsqueeze(0)
        if noise is None:
            raise ValueError('BAYES_EXPLICIT_DRAWS_REQUIRED')
        ew, eb = noise
        w = self.weight + F.softplus(self.weight_rho) * ew
        b = self.bias + F.softplus(self.bias_rho) * eb
        return torch.einsum('bi,sai->sba', x, w) + b[:, None, :]

    def kl(self):
        if not self.bayesian:
            return self.weight.new_zeros(())
        result = self.weight.new_zeros(())
        for mean, rho, prior in ((self.weight, self.weight_rho, self.prior_weight),
                                  (self.bias, self.bias_rho, self.prior_bias)):
            std = F.softplus(rho)
            result = result + (torch.log(self.prior_std / std)
                + (std.square() + (mean-prior).square()) / (2*self.prior_std.square()) - .5).sum()
        return result


def actor_objective(logits, q, alpha):
    """Exact categorical expectation; critics never receive actor gradients."""
    logp = logits.log_softmax(-1)
    return (logp.exp() * (alpha * logp - q.detach())).sum(-1).mean()


def uncertainty(logits):
    logp = logits.log_softmax(-1)
    p = logp.exp()
    mean = p.mean(0)
    entropy = -(mean * mean.clamp_min(1e-30).log()).sum(-1)
    expected = -(p * logp).sum(-1).mean(0)
    return dict(probability=mean, probability_std=p.std(0, correction=0),
                entropy=entropy, expected_entropy=expected,
                mutual_information=(entropy-expected).clamp_min(0))


def hidden(actor, x):
    if actor.core_first_batchnorm:
        encoded = actor.encoder.sequence({'features': x},
            core_first_norm=actor.regularizer).flatten(1)
    else:
        encoded = actor.regularizer(actor.encoder({'features': x}))
    return actor.head[:-1](encoded)
