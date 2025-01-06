import numpy as np
import torch
from torch import linspace, meshgrid, hstack,vstack, zeros, sin,pi
#torch.set_default_dtype(torch.float64)
dtype=torch.float64

def initial_u(ini,x, t=0):

    rho_init = torch.where(x <= 0.5, torch.tensor(ini[0]),torch.tensor(ini[3])).to(torch.float64)
    p_init = torch.where(x <= 0.5, torch.tensor(ini[1]),torch.tensor(ini[4])).to(torch.float64)                                          
    u_init = torch.where(x <= 0.5, torch.tensor(ini[2]),torch.tensor(ini[5])).to(torch.float64)  

    return rho_init,  p_init,u_init

def create_IC_data(ini,Xi, Xf, Ti, Tf, IC_pts):
    x =linspace(Xi, Xf, IC_pts,dtype=dtype)                                   
    t =linspace(Ti, Tf, IC_pts,dtype=dtype)                                     
    t_grid, x_grid = meshgrid(t, x,indexing='xy')                                 
    id_ic = np.random.choice(IC_pts, IC_pts, replace=False)   
    #id_ic=np.arange(0, IC_pts)
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]                                      
    x_ic = x_grid[id_ic, 0][:, None]                                   
    t_ic = t_grid[id_ic, 0][:, None]                                   
    x_ic_train = hstack((t_ic, x_ic)) 
    IC_u = initial_u(ini,x_ic)
    return x_ic_train,IC_u

def create_BC_data(ini,Xi, Xf, Ti, Tf, BC_pts):
    x1 =linspace(Xi, Xi, BC_pts,dtype=dtype)     
    x2 =linspace(Xf, Xf, BC_pts,dtype=dtype)
    t =linspace(Ti, Tf, BC_pts,dtype=dtype)                                                                  
    id_bc = np.random.choice(BC_pts, BC_pts, replace=False)   
    #id_ic=np.arange(0, IC_pts)                                     
    x1_bc = x1[id_bc][:, None]     
    x2_bc = x2[id_bc][:, None]   
    x_bc = vstack((x1_bc,x2_bc))
    u_bc = initial_u(ini,x_bc)
    t_bc = t[id_bc][:, None]                                   
    xi_bc = hstack((t_bc, x1_bc)) 
    xf_bc = hstack((t_bc, x2_bc)) 
    x_bc = vstack((xi_bc,xf_bc))
    return x_bc, u_bc


def create_residual_data(Xi, Xf, Ti, Tf, Nc, N_test,Tc,T_test,N_simple):
    ##########################################################
    x =linspace(Xi, Xf, Nc,dtype=dtype)                                   
    t =linspace(Ti, Tf, Tc,dtype=dtype)                                     
    t_grid, x_grid = meshgrid(t, x,indexing='xy')                                 
    T = t_grid.flatten()[:, None]                                      
    X = x_grid.flatten()[:, None]  

    id_f = np.random.choice(Nc*Tc,N_simple , replace=False)   
    x_int = X[:, 0][id_f, None]                                        
    t_int = T[:, 0][id_f, None]                                        
    xt_resid = hstack((t_int, x_int))
    f_hat_train = zeros((xt_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test,dtype=dtype)
    t_test = linspace(Ti, Tf, T_test,dtype=dtype)
    
    TT_test, XX_test = meshgrid(t_test, x_test,indexing='xy')
    
    X_test = XX_test.flatten()[:,None]
    T_test = TT_test.flatten()[:,None]
    
    xt_test    = hstack((T_test, X_test))
    ##########################################################
    return (xt_resid, f_hat_train, xt_test)

def create_RH_data(Xi, Xf, Ti, Tf, Nc,N_simple,dx):
    x =linspace(Xi, Xf, Nc,dtype=dtype)                                 
    t =linspace(Tf, Tf, Nc,dtype=dtype)                                     

    id_ic = np.random.choice(Nc, N_simple, replace=False)   

    x_RH = x[id_ic][:, None]
    x_RHL = x_RH-dx                                   
    t_ic = t[id_ic][:, None]                                   
    x_RH_train = hstack((t_ic, x_RH)) 
    x_RHL_train = hstack((t_ic, x_RHL))

    return x_RH_train,x_RHL_train

def Pertur_1D(x,T,dt,dx):
  N=x.shape[0]
  xs   = np.zeros((N,2)) 
  xsL  = np.zeros((N,2)) 
  xsR  = np.zeros((N,2)) 
  xsP  = np.zeros((N,2)) 
  xsPL = np.zeros((N,2)) 
  xsPR = np.zeros((N,2)) 
  
  for i in range(N):
      xs[i,1] = x[i,1]
      xs[i,0] = x[i,0] + T
      xsL[i,1] = xs[i,1] - dx
      xsL[i,0] = xs[i,0]
      xsR[i,1] = xs[i,1] + dx
      xsR[i,0] = xs[i,0]
      xsP[i,0] = xs[i,0] + dt
      xsP[i,1] = xs[i,1]
      xsPL[i,0] = xsP[i,0]
      xsPL[i,1] = xsP[i,1]+ dx#feel error
      xsPR[i,0] = xsP[i,0]
      xsPR[i,1] = xsP[i,1]- dx
      
  return xs,xsL,xsR,xsP,xsPL,xsPR

def Move_Time_1D(x,dt):
    N=x.shape[0]
    xen =np.zeros((N,2)) 
    
    for i in range(N):
        xen[i,1] = x[i,1]
        xen[i,0] = x[i,0] + dt
    xen = torch.tensor(xen).to(torch.float64)
    return xen

def Euler_WENO(Xi,Xf,Ti,Tf,ini,gama,delta_t=0.0001,delta_x=0.005):
    #global P,u,C,F_p,F_n,Den,E,FF_p,FF_n,IS1_p,IS1_n,IS2_p,IS2_n,IS3_p,IS3_n
    Nt=int((Tf-Ti)/delta_t)
    Nx=int((Xf-Xi)/delta_x)
    Den=np.array([ini[0]]*int(Nx/2)+[ini[3]]*int(Nx/2))
    P=np.array([ini[1]]*int(Nx/2)+[ini[4]]*int(Nx/2))
    u=np.array([ini[3]]*int(Nx/2)+[ini[5]]*int(Nx/2))
    F_p=np.array([[0.0]*Nx,[0.0]*Nx,[0.0]*Nx])
    F_n=np.array([[0.0]*Nx,[0.0]*Nx,[0.0]*Nx])
    u=np.array([0.0]*Nx)
    Den_u=Den*u
    FF_p=np.array([[0.0]*Nx,[0.0]*Nx,[0.0]*Nx])
    FF_n=np.array([[0.0]*Nx,[0.0]*Nx,[0.0]*Nx])
    E=P/(gama-1)+0.5*u**2*Den
    for j in range(Nt):
        F=[Den_u,Den*u**2+P,u*(E+P)]
        C=np.sqrt(abs(gama*P/Den))
        lamta=np.array([u,u-C,u+C])
        lamta_p=(abs(lamta)+lamta)/2
        lamta_n=lamta-lamta_p
        for i in range(Nx):
            F_p[0][i]=Den[i]/(2*gama)*(2*(gama-1)*lamta_p[0][i]+lamta_p[1][i]+lamta_p[2][i])
            F_p[1][i]=Den[i]/(2*gama)*(2*(gama-1)*lamta_p[0][i]*u[i]+lamta_p[1][i]*(u[i]-C[i])+lamta_p[2][i]*(u[i]+C[i]))
            F_p[2][i]=Den[i]/(2*gama)*((gama-1)*lamta_p[0][i]*u[i]**2+1/2*lamta_p[1][i]*(u[i]-C[i])**2+1/2*lamta_p[2][i]*(u[i]+C[i])**2+(3-gama)/(2*gama-2)*(lamta_p[1][i]+lamta_p[2][i])*(C[i])**2)
            F_n[0][i]=Den[i]/(2*gama)*(2*(gama-1)*lamta_n[0][i]+lamta_n[1][i]+lamta_n[2][i])
            F_n[1][i]=Den[i]/(2*gama)*(2*(gama-1)*lamta_n[0][i]*u[i]+lamta_n[1][i]*(u[i]-C[i])+lamta_n[2][i]*(u[i]+C[i]))
            F_n[2][i]=Den[i]/(2*gama)*((gama-1)*lamta_n[0][i]*u[i]**2+1/2*lamta_n[1][i]*(u[i]-C[i])**2+1/2*lamta_n[2][i]*(u[i]+C[i])**2+(3-gama)/(2*gama-2)*(lamta_n[1][i]+lamta_n[2][i])*(C[i])**2)
        for i in range(Nx-4):
            IS1_p=IS2_p=IS3_p=IS1_n=IS2_n=IS3_n=0
            for k in range(3):
                IS1_p+=1/4*((F_p[k][i]-4*F_p[k][i+1]+3*F_p[k][i+2])**2)+     13/12*((F_p[k][i]  -2*F_p[k][i+1]+F_p[k][i+2])**2)
                IS2_p+=1/4*((F_p[k][i+1]-F_p[k][i+3])**2)+                 13/12*((F_p[k][i+1]-2*F_p[k][i+2]+F_p[k][i+3])**2)
                IS3_p+=1/4*((3*F_p[k][i+2]-4*F_p[k][i+3]+F_p[k][i+4])**2)+   13/12*((F_p[k][i+2]-2*F_p[k][i+3]+F_p[k][i+4])**2)
                IS1_n+=1/4*((F_n[k][i+4]-4*F_n[k][i+3]+3*F_n[k][i+2])**2)+   13/12*((F_n[k][i+4]-2*F_n[k][i+3]+F_n[k][i+2])**2)
                IS2_n+=1/4*((F_n[k][i+3]-F_n[k][i+1])**2)+                 13/12*((F_n[k][i+3]-2*F_n[k][i+2]+F_n[k][i+1])**2)
                IS3_n+=1/4*((3*F_n[k][i+2]-4*F_n[k][i+1]+F_n[k][i])**2)+     13/12*((F_n[k][i+2]-2*F_n[k][i+1]+F_n[k][i])  **2)
            oumiga1_p=(0.1/(IS1_p+1e-6)**2/(0.1/(IS1_p+1e-6)**2+0.6/(IS2_p+1e-6)**2+0.3/(IS3_p+1e-6)**2))#attention,  oumiga is a num instead of an vector
            oumiga2_p=(0.6/(IS2_p+1e-6)**2/(0.1/(IS1_p+1e-6)**2+0.6/(IS2_p+1e-6)**2+0.3/(IS3_p+1e-6)**2))
            oumiga3_p=1-oumiga2_p-oumiga1_p
            oumiga1_n=(0.1/(IS1_n+1e-6)**2/(0.1/(IS1_n+1e-6)**2+0.6/(IS2_n+1e-6)**2+0.3/(IS3_n+1e-6)**2))
            oumiga2_n=(0.6/(IS2_n+1e-6)**2/(0.1/(IS1_n+1e-6)**2+0.6/(IS2_n+1e-6)**2+0.3/(IS3_n+1e-6)**2))
            oumiga3_n=1-oumiga2_n-oumiga1_n
            for k in range(3):
                FF_p[k][i+2]=oumiga1_p*(2*F_p[k][i]/6    -7*F_p[k][i+1]/6   +11*F_p[k][i+2]/6)+oumiga2_p*(-1*F_p[k][i+1]/6 +5*F_p[k][i+2]/6   +2*F_p[k][i+3]/6)+oumiga3_p*(2*F_p[k][i+2]/6  +5*F_p[k][i+3]/6   -1*F_p[k][i+4]/6)
                FF_n[k][i+2]=oumiga1_n*(2*F_n[k][i+4]/6  -7*F_n[k][i+3]/6   +11*F_n[k][i+2]/6)+oumiga2_n*(-1*F_n[k][i+3]/6 +5*F_n[k][i+2]/6   +2*F_n[k][i+1]/6)+oumiga3_n*(2*F_n[k][i+2]/6  +5*F_n[k][i+1]/6     -1*F_n[k][i]/6)
        for i in range(Nx-4):
            if(i>=2):
                Den[i+1]-=   (delta_t/delta_x)*(FF_p[0][i+1]-FF_p[0][i]+FF_n[0][i+2]-FF_n[0][i+1])
                Den_u[i+1]-= (delta_t/delta_x)*(FF_p[1][i+1]-FF_p[1][i]+FF_n[1][i+2]-FF_n[1][i+1])
                E[i+1]-=     (delta_t/delta_x)*(FF_p[2][i+1]-FF_p[2][i]+FF_n[2][i+2]-FF_n[2][i+1])
        u=Den_u/Den
        P=(gama-1)*(E-0.5*Den*u**2)
        x=[[i/Nx] for i in range(Nx)]
    return Den, P, u ,x