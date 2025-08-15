import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from .eos_jwl import jwl_pressure


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

    with torch.no_grad():
        pred = model(xt)
    rho, u, E, lam = [pred[:, i:i+1] for i in range(4)]
    p = jwl_pressure(rho, u, E, cfg["physics"].get("jwl_params", {}))

    nx, nt = X.shape
    rho_t = rho.detach().view(nx, nt).cpu()
    rho = rho_t.numpy()
    u = u.detach().cpu().numpy().reshape(nx, nt)
    p = p.detach().cpu().numpy().reshape(nx, nt)

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

    # Shock trajectory computed as the first crossing of the midpoint
    # density between the left and right states.  This avoids spurious
    # maxima of gradient-based indicators that can pin the shock to a
    # single location.
    rho_left = cfg["ic"]["left"]["rho"]
    rho_right = cfg["ic"]["right"]["rho"]
    rho_mid = 0.5 * (rho_left + rho_right)

    shock_pos = []
    for j in range(nt):
        r = rho_t[:, j]
        # Ignore the boundary point at x=0 which can lead to
        # spurious detections of the shock when the model predicts
        # a low density at the wall.
        mask = r[1:] < rho_mid
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            shock_pos.append(float("nan"))
            continue
        i = idx[0].item() + 1  # offset due to ignoring first point
        x0, x1 = xs[i - 1], xs[i]
        r0, r1 = r[i - 1], r[i]
        if r1 == r0:
            x = x0
        else:
            x = x0 + (rho_mid - r0) * (x1 - x0) / (r1 - r0)
        shock_pos.append(x.item())
    shock_x = torch.tensor(shock_pos, dtype=xs.dtype)
    traj = torch.stack([ts, shock_x], dim=1).cpu().numpy()
    np.savetxt(
        os.path.join(out_dir, "shock_trajectory.csv"),
        traj,
        delimiter=",",
        header="t,x_shock",
        comments="",
    )
    plt.figure()
    plt.plot(ts.cpu().numpy(), shock_x.cpu().numpy())
    plt.xlabel("t")
    plt.ylabel("shock position")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shock_trajectory.png"))
    plt.close()

    return {"rho": rho, "u": u, "p": p}
