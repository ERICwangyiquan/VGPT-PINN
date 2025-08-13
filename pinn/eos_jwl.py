import torch


def jwl_pressure(rho, u, E, params):
    """Compute pressure using a simplified JWL equation of state.

    Parameters
    ----------
    rho : torch.Tensor
        Density.
    u : torch.Tensor
        Velocity.
    E : torch.Tensor
        Total energy density.
    params : dict
        Dictionary with keys A, B, R1, R2, omega.
    """
    A = params.get("A", 1.0)
    B = params.get("B", 1.0)
    R1 = params.get("R1", 4.0)
    R2 = params.get("R2", 1.0)
    omega = params.get("omega", 0.3)

    v = 1.0 / (rho + 1e-12)  # specific volume
    e = E - 0.5 * rho * u ** 2  # internal energy density
    return A * torch.exp(-R1 * v) + B * torch.exp(-R2 * v) + omega * e / v
