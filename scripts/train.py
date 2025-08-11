import argparse
import os
import sys
import yaml

# Ensure repository root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pinn import trainer


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    trainer.train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D Euler PINN")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
