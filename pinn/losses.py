import torch
import torch.autograd as autograd
import torch.nn.functional as F


def euler_residual(model, xt, gamma=1.4):
    """Compute residuals of 1D Euler equations without source terms."""
    xt = xt.requires_grad_(True)
    pred = model(xt)
    rho, u, p = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2

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

    mass_res = rho_t + rho_u_x
    mom_res = rho_u_t + mom_flux_x
    energy_res = E_t + energy_flux_x
    return mass_res, mom_res, energy_res


def pde_loss(model, xt, gamma=1.4):
    mass, mom, energy = euler_residual(model, xt, gamma)
    return (mass ** 2).mean() + (mom ** 2).mean() + (energy ** 2).mean()


def ic_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)


def bc_loss(model, xt, true_u):
    pred = model(xt)
    return F.mse_loss(pred, true_u)
