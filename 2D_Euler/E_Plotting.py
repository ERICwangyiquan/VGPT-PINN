import matplotlib.pyplot as plt
import numpy as np
import torch
import scienceplots
plt.style.use(['science', 'notebook'])

def E_plot(xt, u, scale=150, cmap="rainbow", title=None, dpi=150, figsize=(10,8)):
    
    shape = [int(np.sqrt(u.shape[0])), int(np.sqrt(u.shape[0]))]
    #shape=[100,100]
    
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

def E_plot_2d(x_test,x_grid,y_grid, u, Nd, rx,ry,rd, title = None):
    
    ue = np.zeros((Nd,Nd))
    for j in range(0,Nd):
        for i in range(0,Nd):
            ue[i,j] = u[i*Nd+j]
            x1 = x_test[i*Nd+j,1] 
            y1 = x_test[i*Nd+j,2] 
            if ((x1 - rx)**2 +(y1-ry)**2< rd**2):
                ue[i,j] = 0.0          
    #plt.figure()
    fig, ax = plt.subplots(dpi=80, figsize=(5,4))
    cp =ax.contourf(x_grid[:,0,:],y_grid[:,0,:],ue,60,cmap="rainbow")
    fig.colorbar(cp)
    #ax = plt.gca()
    #ax.set_aspect(1)
        
    if title is not None:
        ax.set_title(title)
    plt.show()
        
    #ax.set_xlabel("$t$", fontsize=25)
    #ax.set_ylabel("$x$", fontsize=25)
        
    #ax.set_xticks([0,0.05,0.1,0.15, 0.2])
    #ax.set_yticks([ 0.0,0.2,0.4,0.6,0.8, 1.0])
    #ax.set_xticks([ 0, 2,  4,6,8, 10])
    #ax.set_yticks([-10.0, -5.0,  0.0,  5.0, 10.0])
    
    #ax.tick_params(axis='both', which='major', labelsize=22.5)
    #ax.tick_params(axis='both', which='minor', labelsize=22.5)



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
