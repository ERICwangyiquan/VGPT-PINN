import torch
import torch.autograd as autograd
import torch.nn.functional as F

from .eos_jwl import jwl_pressure
from .source_terms import arrhenius_rate, energy_source
from .indicators import shock_indicator


def euler_residual(model, xt, cfg):
    """Compute residuals of 1D Euler equations with JWL EOS and source terms."""
    xt = xt.clone().detach().requires_grad_(True)
    pred = model(xt)
    rho, u, E, lam = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]

    p = jwl_pressure(rho, u, E, cfg["physics"].get("jwl_params", {}))

    grad_rho = autograd.grad(rho, xt, torch.ones_like(rho), create_graph=True)[0]
    rho_x, rho_t = grad_rho[:, 0:1], grad_rho[:, 1:2]

    rho_u = rho * u
    grad_rho_u = autograd.grad(rho_u, xt, torch.ones_like(rho_u), create_graph=True)[0]
    rho_u_x, rho_u_t = grad_rho_u[:, 0:1], grad_rho_u[:, 1:2]

    momentum_flux = rho * u ** 2 + p
    grad_mom_flux = autograd.grad(momentum_flux, xt, torch.ones_like(momentum_flux), create_graph=True)[0]
    mom_flux_x = grad_mom_flux[:, 0:1]

    grad_E = autograd.grad(E, xt, torch.ones_like(E), create_graph=True)[0]
    E_x, E_t = grad_E[:, 0:1], grad_E[:, 1:2]

    energy_flux = u * (E + p)
    grad_energy_flux = autograd.grad(energy_flux, xt, torch.ones_like(energy_flux), create_graph=True)[0]
    energy_flux_x = grad_energy_flux[:, 0:1]

    grad_lam = autograd.grad(lam, xt, torch.ones_like(lam), create_graph=True)[0]
    lam_t = grad_lam[:, 1:2]
    rate = arrhenius_rate(rho, u, E, lam, xt, cfg)
    S = energy_source(rho, u, E, lam, xt, cfg)

    mass_res = rho_t + rho_u_x
    mom_res = rho_u_t + mom_flux_x
    energy_res = E_t + energy_flux_x - S
    lambda_res = lam_t - rate
    return mass_res, mom_res, energy_res, lambda_res


def pde_loss(model, xt, cfg):
    mass, mom, energy, lam = euler_residual(model, xt, cfg)
    return (mass ** 2).mean() + (mom ** 2).mean() + (energy ** 2).mean() + (lam ** 2).mean()


def ic_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)


def bc_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)


def shock_loss(model, xt, cfg):
    """Additional loss emphasizing PDE residual near shocks."""
    mass, mom, energy, _ = euler_residual(model, xt, cfg)
    w = shock_indicator(model, xt)
    return (w * (mass**2 + mom**2 + energy**2)).mean()


def rh_loss(model, xt, cfg):
    """Rankine–Hugoniot jump-condition loss at shock locations."""
    w = shock_indicator(model, xt).squeeze()
    thresh = cfg["loss"].get("rh_threshold", 0.0)
    mask = w > thresh
    if mask.sum() == 0:
        return torch.tensor(0.0, device=xt.device)
    xt_shock = xt[mask]
    eps = cfg["loss"].get("rh_epsilon", 1e-3)
    left = xt_shock.clone()
    right = xt_shock.clone()
    left[:, 0] = (left[:, 0] - eps).clamp(min=0.0)
    right[:, 0] = right[:, 0] + eps
    pred_l = model(left)
    pred_r = model(right)
    rho_l, u_l, E_l = pred_l[:, 0:1], pred_l[:, 1:2], pred_l[:, 2:3]
    rho_r, u_r, E_r = pred_r[:, 0:1], pred_r[:, 1:2], pred_r[:, 2:3]
    p_l = jwl_pressure(rho_l, u_l, E_l, cfg["physics"].get("jwl_params", {}))
    p_r = jwl_pressure(rho_r, u_r, E_r, cfg["physics"].get("jwl_params", {}))
    mass_flux_l = rho_l * u_l
    mass_flux_r = rho_r * u_r
    mom_flux_l = rho_l * u_l ** 2 + p_l
    mom_flux_r = rho_r * u_r ** 2 + p_r
    energy_flux_l = u_l * (E_l + p_l)
    energy_flux_r = u_r * (E_r + p_r)
    return (
        (mass_flux_l - mass_flux_r) ** 2
        + (mom_flux_l - mom_flux_r) ** 2
        + (energy_flux_l - energy_flux_r) ** 2
    ).mean()

