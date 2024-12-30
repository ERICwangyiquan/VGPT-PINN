import numpy as np
import torch
import torch.autograd as autograd
from torch import linspace, meshgrid, hstack, zeros, sin, pi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from B_Plotting import Burgers_plot


def initial_u(nu, x, t=0):
    u = nu[1]+nu[0]*sin((2*np.pi*x))
    return u

def exact_u(nu,xt):

    condition = (xt[:,0]-nu*xt[:,1]/2)<= 0.0
    u=torch.where(condition,nu*torch.tensor(1.0), torch.tensor(0.0)).to(torch.float32)

    return u

def create_IC_data(nu, Xi, Xf, Ti, Tf, IC_pts, IC_simple):
    ##########################################################   
    x_IC = linspace(Xi, Xf, IC_pts)
    t_IC = linspace(Ti, Ti, IC_pts)
    X_IC, T_IC = meshgrid(x_IC, t_IC, indexing='ij')

    #id_ic = np.random.choice(IC_pts, IC_simple, replace=False)  

    IC_x = X_IC[:,0][:,None]
    IC_t = zeros(IC_x.shape[0], 1)
    IC_u = initial_u(nu, IC_x)     
    IC   = hstack((IC_x, IC_t))
    return (IC, IC_u)  

def create_BC_data(Xi, Xf, Ti, Tf, BC_pts):
    ##########################################################
    x_BC = linspace(Xi, Xf, BC_pts)
    t_BC = linspace(Ti, Tf, BC_pts)
    X_BC, T_BC = meshgrid(x_BC, t_BC, indexing='ij')

    id_bc = np.random.choice(BC_pts, BC_pts, replace=False)  

    BC_bottom_x = X_BC[0,id_bc][:,None] 
    BC_bottom_t = T_BC[0,id_bc][:,None] 
    BC_bottom   = hstack((BC_bottom_x, BC_bottom_t)) 

    BC_top_x = X_BC[-1,id_bc][:,None] 
    BC_top_t = T_BC[-1,id_bc][:,None] 
    BC_top   = hstack((BC_top_x, BC_top_t))
    return BC_bottom, BC_top

def create_residual_data(Xi, Xf, Ti, Tf, Nx_train, Nt_train, Nx_test, Nt_test,dx,dt):
    ##########################################################
    x_resid = linspace(Xi, Xf, Nx_train)[1:-1]
    t_resid = linspace(Ti, Tf, Nt_train)[1:-1]
    
    XX_resid, TT_resid = meshgrid((x_resid, t_resid), indexing='ij')
    
    X_resid = XX_resid.transpose(1,0).flatten()[:,None]
    T_resid = TT_resid.transpose(1,0).flatten()[:,None]

    id_f =np.random.choice((Nx_train-2)*(Nt_train-2), (Nx_train-2)*(Nt_train-2), replace=False)  
    X_resid = X_resid[:, 0][id_f, None]                                        
    T_resid = T_resid[:, 0][id_f, None]   
    X_RHL = X_resid -dx 
    X_RHR = X_resid +dx 
    xt_resid    = hstack((X_resid, T_resid)) 
    xt_RHL    = hstack((X_RHL, T_resid))
    xt_RHR    = hstack((X_RHR, T_resid))
    xt_RHt    = hstack((X_resid, T_resid+dt)) 
    xt_RHtL    = hstack((X_RHL, T_resid+dt)) 
    f_hat_train = zeros((xt_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, Nx_test)
    t_test = linspace(Ti, Tf, Nt_test)
    XX_test, TT_test = meshgrid((x_test, t_test), indexing='ij')
    
    X_test = XX_test.transpose(1,0).flatten()[:,None]
    T_test = TT_test.transpose(1,0).flatten()[:,None]
    
    xt_test   = hstack((X_test, T_test))
    ##########################################################
    return (xt_resid, f_hat_train, xt_test, xt_RHL,xt_RHR,xt_RHt,xt_RHtL)

def create_RH_data_endt(Xi, Xf, Ti, Tf, Nc,N_simple,dx):
    x =linspace(Xi, Xf, Nc)                                 
    t =linspace(Tf, Tf, Nc)                                     

    id_ic = np.random.choice(Nc, N_simple, replace=False)   

    x_RH = x[id_ic][:, None]
    x_RHL = x_RH-dx                                   
    t_ic = t[id_ic][:, None]                                   
    x_RH_train = hstack((x_RH,t_ic)) 
    x_RHL_train = hstack((x_RHL,t_ic))

    return x_RH_train,x_RHL_train

def create_RH_data(Xi, Xf, Ti, Tf, Nc,N_simple,dx):
    x =linspace(Xi, Xf, Nc)                                 
    t =linspace(Ti, Tf, Nc)                                     
    XX_test, TT_test = meshgrid((x,t), indexing='ij')
    X_test = XX_test.transpose(1,0).flatten()
    T_test = TT_test.transpose(1,0).flatten()

    #id_ic = np.random.choice(Nc**2, N_simple**2, replace=False)   

    #x_RH = X_test[id_ic][:, None]
    #t_ic = T_test[id_ic][:, None]
    x_RH = X_test[:, None]                              
    t_ic = T_test[:, None]
    x_RHL = x_RH-dx                                        
    x_RH_train = hstack((x_RH,t_ic)) 
    x_RHL_train = hstack((x_RHL,t_ic))

    return x_RH_train,x_RHL_train

def Move_Time_1D(x,dt):
    N=x.shape[0]
    xen =np.zeros((N,2)) 
    
    for i in range(N):
        xen[i,0] = x[i,0]
        xen[i,1] = x[i,1] + dt

    xen = torch.tensor(xen)
    return xen

def exact_u_err(u_exact,u_apr,xt,nu):
    u_sub = torch.sub(u_apr, u_exact)
    u_err = torch.sqrt(sum(u_sub**2)/sum(u_exact**2))
    eta = torch.where(abs(xt[:,0]-0.5)<0.02,torch.tensor(0.0),torch.tensor(1.0)).reshape(-1,1)
    u_cor =  torch.sqrt(sum((u_sub*eta)**2)/sum(u_exact**2))
    Burgers_plot(xt, abs(u_sub*eta), 101,201,title=fr"PINN Cor Error $\mu={nu}$")
    return u_err,u_cor   

def burgersDdt_godunov(t, u):
    fu_interface = godunov(u, scheme=2)
    dudt = (fu_interface - np.roll(fu_interface, -1)) / (1/u.size)
    #ro_fu_interface=np.roll(fu_interface, -1)
    #ro_fu_interface[-1] = ro_fu_interface[-2]
    #dudt = (fu_interface-ro_fu_interface) / (1/u.size)
    #print(dudt)
    return dudt

def godunov(u, scheme):
    fu_interface = np.zeros(u.shape,dtype=np.float32)
    if scheme == 2:
        ur = u + (u - np.roll(u, -1)) / 2
        ul = np.roll(u, 1) + (np.roll(u, 1) - np.roll(u, 2)) / 2
        #ul[0]=ul[1]
        #ur[-2]=ur[-1]
        #ul[0]=0.5
        #ur[-1]=0.5
    elif scheme == 1:
        ur = u
        #print(ur)
        ul = np.roll(u, 1)
        ul[0]=ul[1]
        #print(ul)
        #ul[0] = 1
        
    idx_min = np.logical_and(ul < 0, ur > 0)
    idx_lt = np.logical_and(ul <= ur, np.logical_not(idx_min))
    idx_gt = ul > ur
    fu_interface[idx_min] = np.zeros_like(fu_interface[idx_min])
    fu_interface[idx_lt] = np.minimum(ur[idx_lt] ** 2 / 2, ul[idx_lt] ** 2 / 2)
    fu_interface[idx_gt] = np.maximum(ur[idx_gt] ** 2 / 2, ul[idx_gt] ** 2 / 2)
    #print(fu_interface)
    return fu_interface