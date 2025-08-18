# VGPT-PINN

Viscosity-enhanced Generative Pre-Trained Physics-Informed Neural Networks for parameterized nonlinear conservation laws.

## Paper Links
[arXiv](http://arxiv.org/abs/2501.01587) | [ResearchGate](https://www.researchgate.net/publication/387745006_VGPT-PINN_Viscosity-enhanced_Generative_PreTrained_Physics_Informed_Neural_Networks_for_parameterized_nonlinear_conservation_laws)

## VGPT-PINN Architecture
![image](https://github.com/DuktigYajie/VGPT-PINN/blob/main/VGPT-PINN%20Schematic.png)

## Related Work
[CMAME: TGPT-PINN: Nonlinear model reduction with transformed GPT-PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0045782524004547)  |
[YouTube talk](https://www.youtube.com/watch?v=ODA9Po4FVWA)

## Citation
<blockquote style="border-left: 5px solid #ccc; background-color: #f9f9f9; padding: 10px;">
@article{chen2024tgpt,<br>
&nbsp;&nbsp;&nbsp;title={VGPT-PINN: Viscosity-enhanced Generative Pre-Trained Physics Informed Neural Networks for parameterized nonlinear conservation laws},<br>
&nbsp;&nbsp;&nbsp;author={Ji, Yajie and Chen, Yanlai and Xu, Zhenli},<br>
&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2501.01587},<br>
&nbsp;&nbsp;&nbsp;year={2025}<br>
}
</blockquote>

## 1D Explosive Wave PINN
A minimal physics-informed neural network example for the one-dimensional Euler equations is provided under the `pinn/` and `scripts/` directories. Configuration files in `configs/` specify geometry, sampling, physics parameters and training hyperparameters.

### Features
- JWL equation of state, Arrhenius reaction source term and progress variable \(\lambda\) enforcing energy consistency.
- Gradient-based shock indicator with losses `L_shock` and `L_RH` plus adaptive resampling for sharper discontinuities.
- Optional perfectly matched layer (PML) outflow to suppress reflections (`pml.enabled` in the config).
- Batch training utility for multi-seed experiments and robustness evaluation.

### Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the network (short config runs only two epochs):
   ```bash
   python scripts/train.py --config configs/short.yaml
   ```
3. Evaluate a trained model:
   ```bash
   python scripts/eval.py --config configs/short.yaml --model tmp_model.pth
   ```
4. Run multiple seeds and collect results:
   ```bash
   python scripts/batch_train.py --config configs/short.yaml --seeds 0 1 --outdir batch_test
   ```

Outputs such as time histories, pressure fields and shock trajectories are written to the `outputs/` directory.

## Comments
Questions or suggestions are welcome at jiyajie595@sjtu.edu.cn.
