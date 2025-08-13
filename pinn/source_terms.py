import torch

def arrhenius_rate(rho, u, E, lam, xt, cfg):
    """Arrhenius-type reaction rate limited to explosive region."""
    params = cfg["physics"].get("arrhenius_params", {})
    A = float(params.get("A", 0.0))
    Ea = float(params.get("Ea", 0.0))
    Rgas = float(params.get("Rgas", 8.314))
    n = float(params.get("n", 1.0))
    Cv = float(params.get("Cv", 1000.0))
    Lc = cfg["geometry"]["L_charge"]
    x = xt[:, 0:1]
    # Specific internal energy and temperature approximation
    e = E - 0.5 * rho * u ** 2
    T = e / (rho * Cv + 1e-12)
    rate = A * torch.exp(-Ea / (Rgas * (T + 1e-12))) * (1 - lam) ** n
    chi = (x <= Lc).float()
    return rate * chi

def energy_source(rho, u, E, lam, xt, cfg):
    rate = arrhenius_rate(rho, u, E, lam, xt, cfg)
    Q = float(cfg["physics"]["arrhenius_params"].get("Q", 0.0))
    return rho * Q * rate
