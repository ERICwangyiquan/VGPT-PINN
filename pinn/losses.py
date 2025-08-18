import torch
import torch.autograd as autograd
import torch.nn.functional as F

from .eos_jwl import jwl_pressure, jwl_total_energy
from .source_terms import arrhenius_rate, energy_source
from .indicators import shock_indicator


def euler_residual(model, xt, cfg):
    """Compute residuals of 1D Euler equations with JWL EOS and source terms."""
    xt = xt.requires_grad_(True)
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

    # Apply a simple perfectly matched layer (PML) at the right boundary
    pml_cfg = cfg.get("pml", {})
    if pml_cfg.get("enabled", False):
        L = cfg["geometry"]["L_tot"]
        x_start = float(pml_cfg.get("x_start", 0.8 * L))
        sigma_max = float(pml_cfg.get("sigma_max", 50.0))
        x = xt[:, 0:1]
        sigma = torch.zeros_like(x)
        mask = x > x_start
        if mask.any():
            xi = (x[mask] - x_start) / (L - x_start + 1e-12)
            sigma[mask] = sigma_max * xi ** 2

            # Far-field reference state from right IC
            ic_r = cfg.get("ic", {}).get("right", {})
            params = cfg.get("physics", {}).get("jwl_params", {})
            rho_inf = torch.tensor(ic_r.get("rho", 0.0), device=x.device)
            u_inf = torch.tensor(ic_r.get("u", 0.0), device=x.device)
            p_inf = torch.tensor(ic_r.get("p", 0.0), device=x.device)
            E_inf = jwl_total_energy(rho_inf, u_inf, p_inf, params)

            rho_inf = rho_inf.expand_as(rho)
            u_inf = u_inf.expand_as(u)
            E_inf = E_inf.expand_as(E)
            lam_inf = torch.zeros_like(lam)
            rho_u_inf = rho_inf * u_inf

            mass_res += sigma * (rho - rho_inf)
            mom_res += sigma * (rho_u - rho_u_inf)
            energy_res += sigma * (E - E_inf)
            lambda_res += sigma * (lam - lam_inf)

    # Optional scaling to non-dimensionalize residuals and stabilize training
    scales = cfg.get("loss", {}).get("res_scale", {})
    mass_res = mass_res / scales.get("mass", 1.0)
    mom_res = mom_res / scales.get("mom", 1.0)
    energy_res = energy_res / scales.get("energy", 1.0)
    lambda_res = lambda_res / scales.get("lam", 1.0)

    return mass_res, mom_res, energy_res, lambda_res


def pde_loss(model, xt, cfg):
    mass, mom, energy, lam = euler_residual(model, xt, cfg)
    return (mass ** 2).mean() + (mom ** 2).mean() + (energy ** 2).mean() + (lam ** 2).mean()


def shock_loss(model, xt, cfg):
    mass, mom, energy, lam = euler_residual(model, xt, cfg)
    weight = shock_indicator(model, xt)
    weight = weight / (weight.max() + 1e-8)
    res = mass ** 2 + mom ** 2 + energy ** 2 + lam ** 2
    return (weight * res).mean()


def rh_loss(model, xt, cfg, eps=1e-3):
    L = cfg["geometry"]["L_tot"]
    xt_l = xt.clone()
    xt_r = xt.clone()
    xt_l[:, 0] = torch.clamp(xt_l[:, 0] - eps, 0.0, L)
    xt_r[:, 0] = torch.clamp(xt_r[:, 0] + eps, 0.0, L)
    pred_l = model(xt_l)
    pred_r = model(xt_r)
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


def ic_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)


def bc_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)
