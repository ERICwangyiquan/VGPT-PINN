import torch
import torch.autograd as autograd

def shock_indicator(model, xt):
    """Gradient-based shock indicator using density gradient magnitude."""
    xt = xt.requires_grad_(True)
    rho = model(xt)[:, 0:1]
    grad_rho = autograd.grad(rho, xt, torch.ones_like(rho), create_graph=True)[0]
    return grad_rho[:, 0:1].abs()
