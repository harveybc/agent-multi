"""Read-only gradient diagnostics; never update a policy or its optimizer."""
import torch


def gradient_geometry(actor_loss, critic_loss, parameters):
    parameters = list(parameters)
    vectors = []
    for loss in (actor_loss, critic_loss):
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1)
                            for p, g in zip(parameters, grads)]).double()
        if not torch.isfinite(vector).all():
            raise ValueError('NONFINITE_GRADIENT')
        vectors.append(vector)
    a, b = vectors
    na, nb = a.norm().item(), b.norm().item()
    return dict(actor_norm=na, critic_norm=nb,
                cosine=float(torch.dot(a, b)/(na*nb)) if na and nb else None,
                critic_to_actor_norm=nb/na if na else None)
