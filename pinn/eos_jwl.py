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
    A = float(params.get("A", 1.0))
    B = float(params.get("B", 1.0))
    R1 = float(params.get("R1", 4.0))
    R2 = float(params.get("R2", 1.0))
    omega = float(params.get("omega", 0.3))

    v = 1.0 / (rho + 1e-12)  # specific volume
    e = E - 0.5 * rho * u ** 2  # internal energy density
    return A * torch.exp(-R1 * v) + B * torch.exp(-R2 * v) + omega * e / v


def jwl_total_energy(rho, u, p, params):
    """Compute total energy density from primitive variables using JWL EOS.

    This inverts :func:`jwl_pressure` to recover the total energy given
    density, velocity and pressure. It is useful for constructing reference
    states such as the far-field values used by absorbing boundary layers.

    Parameters
    ----------
    rho : torch.Tensor
        Density.
    u : torch.Tensor
        Velocity.
    p : torch.Tensor
        Pressure.
    params : dict
        Dictionary with keys A, B, R1, R2 and omega.
    """
    A = float(params.get("A", 1.0))
    B = float(params.get("B", 1.0))
    R1 = float(params.get("R1", 4.0))
    R2 = float(params.get("R2", 1.0))
    omega = float(params.get("omega", 0.3))

    v = 1.0 / (rho + 1e-12)  # specific volume
    exp1 = torch.exp(-R1 * v)
    exp2 = torch.exp(-R2 * v)
    e = (p - A * exp1 - B * exp2) * v / omega
    return e + 0.5 * rho * u ** 2
