import torch
from .indicators import shock_indicator


def sample_initial(cfg):
    N = cfg["sampling"]["N_ic"]
    L = cfg["geometry"]["L_tot"]
    Lc = cfg["geometry"]["L_charge"]
    gamma = cfg["physics"].get("gamma", 1.4)
    left = cfg["ic"]["left"]
    right = cfg["ic"]["right"]
    x = torch.rand(N, 1) * L
    t = torch.zeros_like(x)
    rho = torch.where(x <= Lc, torch.full_like(x, left["rho"]), torch.full_like(x, right["rho"]))
    u = torch.where(x <= Lc, torch.full_like(x, left["u"]), torch.full_like(x, right["u"]))
    p = torch.where(x <= Lc, torch.full_like(x, left["p"]), torch.full_like(x, right["p"]))
    E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    lam = torch.where(x <= Lc, torch.zeros_like(x), torch.ones_like(x))
    xt = torch.cat([x, t], dim=1)
    u_vec = torch.cat([rho, u, E, lam], dim=1)
    return xt, u_vec


def sample_boundary(cfg):
    N = cfg["sampling"]["N_bc"]
    L = cfg["geometry"]["L_tot"]
    Lc = cfg["geometry"]["L_charge"]
    T = cfg["time"]["T_end"]
    gamma = cfg["physics"].get("gamma", 1.4)
    left = cfg["ic"]["left"]
    right = cfg["ic"]["right"]
    t = torch.rand(N, 1) * T

    x0 = torch.zeros_like(t)
    xt_left = torch.cat([x0, t], dim=1)
    E_left = left["p"] / (gamma - 1.0) + 0.5 * left["rho"] * left["u"] ** 2
    u_left = torch.tensor([left["rho"], left["u"], E_left, 0.0]).repeat(N, 1)

    xL = torch.full_like(t, L)
    xt_right = torch.cat([xL, t], dim=1)
    E_right = right["p"] / (gamma - 1.0) + 0.5 * right["rho"] * right["u"] ** 2
    lam_right = 1.0 if L > Lc else 0.0
    u_right = torch.tensor([right["rho"], right["u"], E_right, lam_right]).repeat(N, 1)
    return (xt_left, u_left), (xt_right, u_right)


def sample_residual(cfg):
    N = cfg["sampling"]["N_f"]
    L = cfg["geometry"]["L_tot"]
    T = cfg["time"]["T_end"]
    x = torch.rand(N, 1) * L
    t = torch.rand(N, 1) * T
    return torch.cat([x, t], dim=1)


def resample_shock_points(model, cfg):
    N = cfg["sampling"].get("N_shock", 0)
    if N <= 0:
        return None
    device = next(model.parameters()).device
    L = cfg["geometry"]["L_tot"]
    T = cfg["time"]["T_end"]
    N_candidates = N * 5
    x = torch.rand(N_candidates, 1, device=device) * L
    t = torch.rand(N_candidates, 1, device=device) * T
    xt = torch.cat([x, t], dim=1)
    ind = shock_indicator(model, xt).detach().squeeze()
    topk = torch.topk(ind, k=N, largest=True).indices
    return xt[topk].detach()


def sample_training_points(cfg):
    ic_xt, ic_u = sample_initial(cfg)
    (bc_left_xt, bc_left_u), (bc_right_xt, bc_right_u) = sample_boundary(cfg)
    f_xt = sample_residual(cfg)
    return {
        "ic_xt": ic_xt,
        "ic_u": ic_u,
        "bc_left_xt": bc_left_xt,
        "bc_left_u": bc_left_u,
        "bc_right_xt": bc_right_xt,
        "bc_right_u": bc_right_u,
        "f_xt": f_xt,
    }
