import argparse
import os
import sys
import yaml
import torch

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pinn import networks, evaluate


def main(cfg_path, model_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model = networks.MLP(out_dim=4, hidden_layers=cfg["model"]["mlp_hidden"], activation=cfg["model"]["activation"])
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    evaluate.evaluate(model, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 1D Euler PINN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="model.pth")
    args = parser.parse_args()
    main(args.config, args.model)
