import numpy as np
import torch
from torch import linspace, meshgrid, hstack, zeros, sin,pi

def IC(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    rho1 = 2.112
    p1 =  3.011
    v1 = 0.0
    u1 = np.sqrt(1.4*p1/rho1)*0.728
    
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] =   u1
        v_init[i] =  v1
        p_init[i] =  p1
    return rho_init, u_init, v_init,p_init

def BC_L(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))                                                # p - initial condition
    
    gamma = 1.4
    #u1 = ms*npsqrt(gamma)
    # rho, p - initial condition
    rho1 = 2.112
    p1 =  3.011
    v1 = 0.0
    u1 = np.sqrt(1.4*p1/rho1)*0.728
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] =  u1
        v_init[i] =  v1
        p_init[i] =  p1
    return rho_init, u_init, v_init,p_init

def BC_R(x):
    N =x.shape[0]
    rho_init = np.zeros((x.shape[0]))                                              # rho - initial condition
    u_init = np.zeros((x.shape[0]))                                                # u - initial condition
    v_init = np.zeros((x.shape[0]))                                                # u - initial condition
    p_init = np.zeros((x.shape[0]))  

    gamma = 1.4
    ms = 2.0
    rho1 = 1.0
    p1 = 1.0
    v1 = 0.0
    u1 = 0
    # rho, p - initial condition
    for i in range(N):
        rho_init[i] = rho1
        u_init[i] = u1
        v_init[i] = v1
        p_init[i] = p1

    return rho_init, u_init, v_init,p_init

def BD_circle(t,xc,yc,r,n):
    x = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*r
        x[i,2] = yc  + yd*r
        cos[i,0] = xd 
        sin[i,0] = yd
    return x, sin,cos



'''
def initial_u(x,nu, t=0):
    rho_init = 2.112*torch.ones(x.shape[0],1).to(torch.float32)
    p_init = 3.011*torch.ones(x.shape[0],1).to(torch.float32)
    u_init = 1.028*torch.ones([x.shape[0],1]).to(torch.float32)
    v_init = torch.zeros([x.shape[0],1]).to(torch.float32)
    return rho_init,  p_init,u_init,v_init

def create_IC_data(nu,Xi, Xf, Ti, Tf, IC_pts,IC_simples):
    x_IC = linspace(Xi, Xf, IC_pts)
    y_IC = linspace(Xi, Xf, IC_pts)
    t_IC = linspace(Ti, Ti, IC_pts)
    T_IC,X_IC, Y_IC = meshgrid(t_IC,x_IC, y_IC,indexing='ij')
    id_ic = np.random.choice(IC_pts**3, IC_simples, replace=False) 

    X = X_IC.flatten()[:, None]                                      
    IC_x = X[id_ic, 0][:, None] 
    Y = Y_IC.flatten()[:, None]                                      
    IC_y = Y[id_ic, 0][:, None] 
    IC_t = zeros(IC_simples, 1)
    IC_u = initial_u(nu, IC_x, IC_y)     
    IC   = hstack((IC_t,IC_x, IC_y))
    return (IC, IC_u)  

def create_BCL_data(Xi, Xf, Ti, Tf, Nc, N_test,Tc,T_test,N_simple):

def create_BCR_data(Xi, Xf, Ti, Tf, Nc, N_test,Tc,T_test,N_simple):

def BD_circle(t,xc,yc,r,n):
    x = np.zeros((n,3)) 
    sin = np.zeros((n,1)) 
    cos = np.zeros((n,1)) 

    for i in range(n):
        the = 2*np.random.rand()*np.pi
        xd = np.cos(the + np.pi/2)
        yd = np.sin(the + np.pi/2)
        x[i,0] = np.random.rand()*t
        x[i,1] = xc  + xd*r
        x[i,2] = yc  + yd*r
        cos[i,0] = xd 
        sin[i,0] = yd
    return x, sin,cos


def create_residual_data(Xi, Xf, Ti, Tf, Nc, N_test,Tc,T_test,N_simple):
    ##########################################################
    x_resid = linspace(Xi, Xf, Nc)[1:-1]
    y_resid = linspace(Xi, Xf, Nc)[1:-1]
    t_resid = linspace(Ti, Tf, Tc)[1:-1]
    
    TT_resid,XX_resid, YY_resid = meshgrid(t_resid,x_resid, y_resid, indexing='ij')
    id_f = np.random.choice(Nc*Nc*Tc,N_simple , replace=False)  

    X_resid = XX_resid.transpose(1,0).flatten()[:,None]
    Y_resid = YY_resid.transpose(1,0).flatten()[:,None]
    T_resid = TT_resid.transpose(1,0).flatten()[:,None]
    
    X= X_resid[id_f, 0][:, None]
    Y= Y_resid[id_f, 0][:, None]
    T= T_resid[id_f, 0][:, None]
    xt_resid    = hstack((T,X,Y)) 
    f_hat_train = zeros((xt_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test)
    y_test = linspace(Xi, Xf, N_test)
    t_test = linspace(Ti, Tf, T_test)
    XX_test, YY_test, TT_test = meshgrid(x_test, y_test, t_test,indexing='ij')
    
    X_test = XX_test.transpose(1,0).flatten()[:,None]
    Y_test = YY_test.transpose(1,0).flatten()[:,None]
    T_test = TT_test.transpose(1,0).flatten()[:,None]
    
    xt_test   = hstack((T_test,X_test, Y_test))
    ##########################################################
    return (xt_resid, f_hat_train, xt_test)
'''
