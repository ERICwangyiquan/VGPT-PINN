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
        # find the first interior cell where the density crosses the midpoint
        above = r >= rho_mid
        # skip boundary cells to avoid spurious matches at x≈0 or x≈L
        crossing = above[1:-1] & ~above[2:]
        idx = torch.nonzero(crossing, as_tuple=False)
        if idx.numel() == 0:
            # fall back to maximum density gradient away from boundaries
            grad = r[1:] - r[:-1]
            if grad.numel() < 3:
                shock_pos.append(float("nan"))
                continue
            i = grad[1:-1].abs().argmax().item() + 1
            shock_pos.append(xs[i].item())
            continue
        i = idx[0].item() + 2  # crossing occurs between i-1 and i
        x0, x1 = xs[i - 1], xs[i]
        r0, r1 = r[i - 1], r[i]
        if r1 == r0:
            # fallback: use interior maximum gradient for a sub-cell estimate
            grad = r[1:] - r[:-1]
            if grad.numel() < 3:
                shock_pos.append(float("nan"))
            else:
                i = grad[1:-1].abs().argmax().item() + 1
                shock_pos.append(xs[i].item())
        else:
            x = x0 + (rho_mid - r0) * (x1 - x0) / (r1 - r0)
            shock_pos.append(x.item())
    # enforce non-decreasing shock position
    for k in range(1, len(shock_pos)):
        if shock_pos[k] < shock_pos[k - 1]:
            shock_pos[k] = shock_pos[k - 1]
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
