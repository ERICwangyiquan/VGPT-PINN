import torch
import torch.autograd as autograd

def shock_indicator(model, xt):
    """Gradient-based shock indicator using density gradient magnitude."""
    # Ensure xt is a leaf tensor with grad enabled to avoid zeros from autograd
    xt = xt.clone().detach().requires_grad_(True)
    rho = model(xt)[:, 0:1]
    grad_rho = autograd.grad(rho, xt, torch.ones_like(rho), create_graph=False)[0]
    ind = grad_rho[:, 0:1].abs()
    # Replace any NaN/Inf values that can arise from ill-conditioned gradients
    return torch.nan_to_num(ind, nan=0.0, posinf=0.0, neginf=0.0)
