"""Read-only action ranking and first-step shared-head AdamW diagnostics."""
import torch


def adaptive_parts(part, weights, margin_jacobian, theta, lr, decay, eps):
    g = part['batch_gradient']
    scale = lr / (g.abs() + eps)
    own = -torch.einsum('ndp,np,p->nd', margin_jacobian,
                       part['per_row_gradient'], scale) * weights[:, None]
    total = -torch.einsum('ndp,p,p->nd', margin_jacobian, g, scale)
    wd = margin_jacobian @ (-lr * decay * theta)
    return dict(self=own, cross=total-own, adaptive=total, decay=wd, total=total+wd)


def action_quality(native, q):
    chosen = q.argmax(-1)
    regret = native.max(-1).values - native.gather(1, chosen[:, None]).squeeze(1)
    gap = native[:, 1:] - native[:, :1]
    qgap = q[:, 1:] - q[:, :1]
    positive = gap > 1e-8
    top = native.topk(2, dim=-1).values
    return dict(rows=len(native), optimal_count=int((regret <= 1e-8).sum()),
                unique_native_optimum=int(((top[:, 0]-top[:, 1]) > 1e-8).sum()),
                mean_regret=float(regret.mean()) if len(regret) else None,
                beneficial_gaps=int(positive.sum()),
                beneficial_gap_sign_errors=int((positive & (qgap <= 0)).sum()),
                predicted_actions=torch.bincount(chosen, minlength=3).tolist(),
                native_actions=torch.bincount(native.argmax(-1), minlength=3).tolist())
