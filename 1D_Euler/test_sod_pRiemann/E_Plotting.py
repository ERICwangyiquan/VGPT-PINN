import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
plt.style.use(['science', 'notebook'])

def E_plot(xt, u, scale=150, cmap="rainbow", title=None, dpi=150, figsize=(10,8)):
    
    #shape = [int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0]))]
    shape=[100,100]
    
    x = xt[:,1].reshape(shape=shape).transpose(1,0).cpu().detach() 
    t = xt[:,0].reshape(shape=shape).transpose(1,0).cpu().detach() 
    u =       u.reshape(shape=shape).transpose(1,0).cpu().detach()
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    cp = ax.contourf(t, x, u, scale, cmap=cmap)
    fig.colorbar(cp)
        
    #ax.set_xlabel("$t$", fontsize=25)
    #ax.set_ylabel("$x$", fontsize=25)
        
    #ax.set_xticks([0,0.05,0.1,0.15, 0.2])
    #ax.set_yticks([ 0.0,0.2,0.4,0.6,0.8, 1.0])
    #ax.set_xticks([ 0, 2,  4,6,8, 10])
    #ax.set_yticks([-10.0, -5.0,  0.0,  5.0, 10.0])
    
    #ax.tick_params(axis='both', which='major', labelsize=22.5)
    #ax.tick_params(axis='both', which='minor', labelsize=22.5)
    
    if title is not None:
        ax.set_title(title)#, fontsize=25
        
    plt.show()

def E_plot_ut(xt_test,TGPT_sol,weno_u,Nx=100,ind = 100):

    plt.figure(dpi=150, figsize=(5,4))
    plt.plot(weno_u[3],weno_u[0],'k')#,label="$Density$"
    plt.plot(weno_u[3],weno_u[1],'k')#,label="$Pressure$"
    plt.plot(weno_u[3],weno_u[2],'k')#,label="$velocity$"
    plt.plot(xt_test[0::Nx,1].detach().cpu(),TGPT_sol[0][ind-1::Nx].detach().cpu(),'--',label=fr"$Density$")
    plt.plot(xt_test[0::Nx,1].detach().cpu(),TGPT_sol[1][ind-1::Nx].detach().cpu(),'--',label=fr"$Pressure$")
    plt.plot(xt_test[0::Nx,1].detach().cpu(),TGPT_sol[2][ind-1::Nx].detach().cpu(),'--',label=fr"$Velocity$")
    plt.legend(fontsize = 12)
    plt.show()


def loss_plot(epochs, losses, title=None, dpi=150, figsize=(10,8)):
    """Training losses"""
    plt.figure(dpi=dpi, figsize=figsize)
    plt.plot(epochs, losses, c="k", linewidth=3)
    
    #plt.xlabel("Epoch",     fontsize=22.5)
    #plt.ylabel("Loss", fontsize=22.5)
     
    plt.grid(True)
    plt.xlim(0,max(epochs))
    plt.yscale('log')
    
    if title is not None:
        plt.title(title)
    
    plt.show() 
