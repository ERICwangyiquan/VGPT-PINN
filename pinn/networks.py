import torch
import torch.nn as nn

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
    """Simple fully-connected network mapping (x,t) -> (rho,u,p)."""

    def __init__(self, in_dim=2, out_dim=3, hidden_layers=None, activation="tanh"):
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
        return self.model(x)
