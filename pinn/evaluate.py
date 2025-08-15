import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .eos_jwl import jwl_pressure
from .indicators import shock_indicator


def evaluate(model, cfg, device=None, out_dir="outputs"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    L = cfg["geometry"]["L_tot"]
    T = cfg["time"]["T_end"]
    dt = cfg["time"]["dt_vis"]
    x_obs = cfg.get("evaluation", {}).get("x_obs", L / 2)

    xs = torch.linspace(0, L, 1000)
    ts = torch.arange(0, T + 1e-12, dt)
    X, Tm = torch.meshgrid(xs, ts, indexing="ij")
    xt = torch.stack([X.reshape(-1), Tm.reshape(-1)], dim=1).to(device)

    xt = xt.requires_grad_(True)
    pred = model(xt)
    ind = shock_indicator(model, xt)
    rho, u, E, lam = [pred[:, i:i+1] for i in range(4)]
    p = jwl_pressure(rho, u, E, cfg["physics"].get("jwl_params", {}))

    nx, nt = X.shape
    rho = rho.detach().cpu().numpy().reshape(nx, nt)
    u = u.detach().cpu().numpy().reshape(nx, nt)
    p = p.detach().cpu().numpy().reshape(nx, nt)
    # Clamp NaN/Inf values in the indicator to avoid argmax returning index 0
    ind = torch.nan_to_num(ind.detach(), nan=0.0, posinf=0.0, neginf=0.0).view(nx, nt)

    os.makedirs(out_dir, exist_ok=True)
    # Time history at observation point
    x_obs_t = torch.full((len(ts), 1), x_obs)
    xt_obs = torch.cat([x_obs_t, ts.unsqueeze(1)], dim=1).to(device)
    with torch.no_grad():
        pred_obs = model(xt_obs)
    rho_o, u_o, E_o, lam_o = [pred_obs[:, i:i+1] for i in range(4)]
    p_o = jwl_pressure(rho_o, u_o, E_o, cfg["physics"].get("jwl_params", {}))
    data = torch.cat([ts.unsqueeze(1), rho_o, u_o, p_o], dim=1).cpu().numpy()
    np.savetxt(os.path.join(out_dir, "time_history.csv"), data, delimiter=",", header="t,rho,u,p", comments="")

    # Simple visualization of pressure field
    plt.figure()
    plt.pcolormesh(ts.cpu().numpy(), xs.cpu().numpy(), p, shading="auto")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(label="p")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pressure_field.png"))
    plt.close()

    # Shock trajectory
    val, idx = ind.max(dim=0)
    shock_x = torch.where(val > 0, xs[idx], torch.full_like(val, float("nan")))
    traj = torch.stack([ts, shock_x], dim=1).cpu().numpy()
    np.savetxt(os.path.join(out_dir, "shock_traj.csv"), traj, delimiter=",", header="t,x_shock", comments="")
    plt.figure()
    plt.plot(ts.cpu().numpy(), shock_x.cpu().numpy())
    plt.xlabel("t")
    plt.ylabel("shock position")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shock_traj.png"))
    plt.close()

    return {"rho": rho, "u": u, "p": p}
