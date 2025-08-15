import torch
import torch.autograd as autograd

def shock_indicator(model, xt):
    """Gradient-based shock indicator.

    Parameters
    ----------
    model : torch.nn.Module
        PINN model returning [rho, u, E, lam].
    xt : torch.Tensor, shape (N,2)
        Coordinates where indicator is evaluated.

    Returns
    -------
    torch.Tensor, shape (N,1)
        Magnitude of density gradient |d rho / d x| used as indicator.
    """
    xt = xt.requires_grad_(True)
    rho = model(xt)[:, 0:1]
    grad = autograd.grad(rho, xt, torch.ones_like(rho), create_graph=False)[0]
    rho_x = grad[:, 0:1]
    return rho_x.abs()
