import matplotlib.pyplot as plt
import numpy as np
import scienceplots
plt.style.use(['science', 'notebook'])

def Burgers_plot(xt, u, Nx,Nt, scale=150, cmap="rainbow", title=None, 
                 dpi=80, figsize=(5,4)):#dpi=150, figsize=(10,8)
    """Burgers Contour Plot"""
    
    shape = [Nt, Nx]
    
    x = xt[:,0].reshape(shape=shape).transpose(1,0).cpu().detach() 
    t = xt[:,1].reshape(shape=shape).transpose(1,0).cpu().detach() 
    u =       u.reshape(shape=shape).transpose(1,0).cpu().detach()
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    cp = ax.contourf(t, x, u, scale, cmap=cmap)
    fig.colorbar(cp)
        
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$x$", fontsize=12)
        
    #ax.set_xticks([ 0.0,0.3, 0.6,0.9,1.2,1.5])
    #ax.set_yticks([ -1.0,-0.5, 0.0, 0.5,1.0,1.5])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    
    if title is not None:
        ax.set_title(title, fontsize=12)
        
    plt.show()

def loss_plot(epochs, losses, title=None, dpi=80, figsize=(5,4)):
    """Training losses"""
    plt.figure(dpi=dpi, figsize=figsize)

    plt.plot(epochs, losses, c="k", linewidth=3)
    
    plt.xlabel("Epoch",     fontsize=12)
    plt.ylabel("PINN Loss", fontsize=12)
     
    plt.grid(True)
    plt.xlim(0,max(epochs))
    plt.yscale('log')
    
    if title is not None:
        plt.title(title)
    
    plt.show() 