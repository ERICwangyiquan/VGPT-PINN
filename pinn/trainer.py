import torch
from .networks import MLP
from .losses import pde_loss, ic_loss, bc_loss
from . import sampling


def train(cfg, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(out_dim=4, hidden_layers=cfg["model"]["mlp_hidden"], activation=cfg["model"]["activation"]).to(device)

    data = sampling.sample_training_points(cfg)
    for k in data:
        data[k] = data[k].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    epochs = cfg["train"]["epochs"]

    for ep in range(epochs):
        opt.zero_grad()
        loss_pde = pde_loss(model, data["f_xt"], cfg)
        loss_ic = ic_loss(model, data["ic_xt"], data["ic_u"])
        bc_left = bc_loss(model, data["bc_left_xt"], data["bc_left_u"])
        bc_right = bc_loss(model, data["bc_right_xt"], data["bc_right_u"])
        loss_bc = bc_left + bc_right
        loss = (
            cfg["loss"]["w_pde"] * loss_pde
            + cfg["loss"]["w_ic"] * loss_ic
            + cfg["loss"]["w_bc"] * loss_bc
        )
        loss.backward()
        opt.step()
        if (ep + 1) % 100 == 0:
            print(f"Epoch {ep+1}/{epochs}: loss={loss.item():.4e}")
            
    save_path = cfg["train"].get("save_path")
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model
