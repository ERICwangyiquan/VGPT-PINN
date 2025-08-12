import torch


def sample_initial(cfg):
    N = cfg["sampling"]["N_ic"]
    L = cfg["geometry"]["L_tot"]
    Lc = cfg["geometry"]["L_charge"]
    left = cfg["ic"]["left"]
    right = cfg["ic"]["right"]
    x = torch.rand(N, 1) * L
    t = torch.zeros_like(x)
    rho = torch.where(x <= Lc, torch.full_like(x, left["rho"]), torch.full_like(x, right["rho"]))
    u = torch.where(x <= Lc, torch.full_like(x, left["u"]), torch.full_like(x, right["u"]))
    p = torch.where(x <= Lc, torch.full_like(x, left["p"]), torch.full_like(x, right["p"]))
    xt = torch.cat([x, t], dim=1)
    u_vec = torch.cat([rho, u, p], dim=1)
    return xt, u_vec


def sample_boundary(cfg):
    N = cfg["sampling"]["N_bc"]
    L = cfg["geometry"]["L_tot"]
    T = cfg["time"]["T_end"]
    left = cfg["ic"]["left"]
    right = cfg["ic"]["right"]
    t = torch.rand(N, 1) * T

    x0 = torch.zeros_like(t)
    xt_left = torch.cat([x0, t], dim=1)
    u_left = torch.tensor([left["rho"], left["u"], left["p"]]).repeat(N, 1)

    xL = torch.full_like(t, L)
    xt_right = torch.cat([xL, t], dim=1)
    u_right = torch.tensor([right["rho"], right["u"], right["p"]]).repeat(N, 1)
    return (xt_left, u_left), (xt_right, u_right)


def sample_residual(cfg):
    N = cfg["sampling"]["N_f"]
    L = cfg["geometry"]["L_tot"]
    T = cfg["time"]["T_end"]
    x = torch.rand(N, 1) * L
    t = torch.rand(N, 1) * T
    return torch.cat([x, t], dim=1)


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
