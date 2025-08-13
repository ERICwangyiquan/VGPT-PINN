import torch
import torch.nn as nn
import torch.nn.functional as F

class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "swish": nn.SiLU,
    "sine": Sine,
}

class MLP(nn.Module):
    """Simple fully-connected network mapping (x,t) -> flow variables."""

    def __init__(self, in_dim: int = 2, out_dim: int = 4, hidden_layers=None, activation: str = "tanh"):
        """
        Parameters
        ----------
        in_dim : int
            Dimension of input coordinates (default 2 for x,t).
        out_dim : int
            Dimension of output state. M2 uses [rho, u, E, lambda].
        hidden_layers : list[int]
            Sizes of hidden layers.
        activation : str
            Activation name from ``ACTIVATIONS``.
        """
        super().__init__()
        hidden_layers = hidden_layers or [128, 128, 128]
        act_cls = ACTIVATIONS.get(activation, nn.Tanh)
        layers = []
        last_dim = in_dim
        for h in hidden_layers:
            layers.append(nn.Linear(last_dim, h))
            layers.append(act_cls())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.model(x)
        raw_rho, u, raw_E, raw_lam = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3], raw[:, 3:4]
        rho = F.softplus(raw_rho)
        E = F.softplus(raw_E)
        lam = torch.sigmoid(raw_lam)
        return torch.cat([rho, u, E, lam], dim=1)
