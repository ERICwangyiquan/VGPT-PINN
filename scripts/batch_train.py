import argparse
import copy
import os
import sys
import yaml
import torch

# Ensure repository root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pinn import trainer


def main(cfg_path, seeds, out_dir):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        cfg_run = copy.deepcopy(cfg)
        cfg_run["train"]["save_path"] = os.path.join(out_dir, f"model_seed{seed}.pth")
        _, loss = trainer.train(cfg_run)
        results.append((seed, loss))
    with open(os.path.join(out_dir, "batch_results.csv"), "w") as f:
        f.write("seed,loss\n")
        for s, l in results:
            f.write(f"{s},{l}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch PINN experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--outdir", default="batch_runs")
    args = parser.parse_args()
    main(args.config, args.seeds, args.outdir)
