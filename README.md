# VGPT-PINN：Viscosity-enhanced Generative Pre-Trained Physics Informed Neural Networks for parameterized nonlinear conservation laws
We propose a Viscosity-enhanced Generative Pre-Trained Physics-Informed Neural Network with a transform layer (VGPT-PINN) for solving parameterized nonlinear conservation laws. The VGPT-PINN extends the traditional physics-informed neural networks and its recently proposed generative pre-trained strategy for linear model reduction to nonlinear model reduction and shock-capturing domains. By utilizing an adaptive meta-network, a simultaneously trained transform layer, viscosity enhancement strategies, implementable shock interaction analysis, and a separable training process, the VGPT-PINN efficiently captures complex parameter-dependent shock formations and interactions.  Numerical results of VGPT-PINN applied to the families of inviscid Burgers' equation and the Euler equations, parameterized by their initial conditions, demonstrate the robustness and accuracy of the proposed technique. It accurately solves for the viscosity solution via very few neurons without leveraging any {\it a priori} knowledge of the equations or its initial condition. 

# Paper Links:
[arXiv](http://arxiv.org/abs/2501.01587) | [ResearchGate](https://www.researchgate.net/publication/387745006_VGPT-PINN_Viscosity-enhanced_Generative_PreTrained_Physics_Informed_Neural_Networks_for_parameterized_nonlinear_conservation_laws)

# VGPT-PINN Architecture:
![image](https://github.com/DuktigYajie/VGPT-PINN/blob/main/VGPT-PINN%20Schematic.png)

# Related Work:
Transformed Generative Pre-Trained Physics-Informed Neural Networks (TGPT-PINN), a framework that extends Physics-Informed Neural Networks (PINNs) and reduced basis methods (RBM) to the non- linear model reduction regime while maintaining the type of network structure and the unsupervised nature of its learning. 

Paper Links:
[CMAME:TGPT-PINN: Nonlinear model reduction with transformed GPT-PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0045782524004547)

Talk/Presentation:
[YouTube](https://www.youtube.com/watch?v=ODA9Po4FVWA)


# Citation:
Below you can find the Bibtex citation:

<blockquote style="border-left: 5px solid #ccc; background-color: #f9f9f9; padding: 10px;">
@article{chen2024tgpt,<br>
&nbsp;&nbsp;&nbsp;title={VGPT-PINN: Viscosity-enhanced Generative Pre-Trained Physics Informed Neural Networks for parameterized nonlinear conservation laws},<br>
&nbsp;&nbsp;&nbsp;author={Ji, Yajie and Chen, Yanlai and Xu, Zhenli},<br>
&nbsp;&nbsp;&nbsp;journal={arXiv preprint arXiv:2501.01587},<br>
&nbsp;&nbsp;&nbsp;year={2025}<br>
}
</blockquote>

# Comments：
Typically, you can directly run the xxx_PINN.ipynb file to generate the full PINN network and save it, then run the xxx_VGPT.ipynb code. If you want to use an existing xxx.pkl file to directly test the VGPT-PINN results, you need to rename the XXX_VGPT_activation.py file to XXX_GPT_activation.py and import XXX_GPT_activation in the XXXX_VGPT.ipynb file. If you have any further questions, feel free to contact me at jiyajie595@sjtu.edu.cn.
