import matplotlib.pyplot as plt
import torch
#from torch._C import _dump_upgraders_map
import torch.autograd as autograd
import torch.nn as nn
#torch.set_default_dtype(torch.float64)
#dtype=torch.float64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)

def source_term(state):
    """Source term :math:`s(u)` for the 1D Euler system.

    Args:
        state (torch.Tensor): Tensor with components ``[rho, p, u]``.

    Returns:
        torch.Tensor: Source contribution for each conservation equation.
        Defaults to zeros, yielding the homogeneous Euler equations.
    """
    return torch.zeros_like(state)

class PINNs_WE_Euler_1D(nn.Module):  
    def __init__(self,Nl,Nn):
        super(PINNs_WE_Euler_1D, self).__init__()
        self.net = nn.Sequential()                                                 
        self.net.add_module('Linear_layer_1', nn.Linear(2, Nn))                    
        self.net.add_module('Tanh_layer_1', nn.Tanh())                             

        for num in range(2, Nl):                                                    
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(Nn, Nn))      
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())                
        self.net.add_module('Linear_layer_final', nn.Linear(Nn, 3))    

    def forward(self, x):       
        a = self.net(x)
        return a
    
    def loss_pde(self, x):
        y = self.forward(x)
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        #rho,p,u = y[0], y[1], y[2]

        # Source terms
        src = source_term(torch.cat([rho, p, u], dim=1))
        s1, s2, s3 = src[:, 0:1], src[:, 1:2], src[:, 2:]

        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4

        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)
        
        gamma = 1.4                                                    

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            

       # dp_g = gradients(p, x)[0]                                     
       # p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                           
        
        dU2_g = gradients(U2, x)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = gradients(U3, x)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = gradients(F2, x)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = gradients(F3, x)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]
        d = 1/(0.2*(abs(u_x)-(u_x) )+1)
        
        eta = (torch.clamp(abs(u_x)-(u_x)-1,min=0)*torch.clamp(abs(p_x)-(p_x)-1,min=0)).detach()
        d = 1/(eta*(abs(u_x)-(u_x) )+1)
     
        f = ((d*(rho_t + U2_x - s1))**2).mean() + \
            ((d*(U2_t  + F2_x - s2))**2).mean() + \
            ((d*(U3_t  + F3_x - s3))**2).mean() #+\
            #((rho_t).mean())**2 +\
            #((U3_t).mean())**2
    
        return f

    def loss_pde2(self, x):
        y = self.forward(x)
        rho,p,u = y[:, 0:1], y[:, 1:2], y[:, 2:]
        #rho,p,u = y[0], y[1], y[2]

        # Source terms
        src = source_term(torch.cat([rho, p, u], dim=1))
        s1, s2, s3 = src[:, 0:1], src[:, 1:2], src[:, 2:]

        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/0.4

        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)
        
        gamma = 1.4                                                    

        # Gradients and partial derivatives
        drho_g = gradients(rho, x)[0]                                  
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g = gradients(u, x)[0]                                      
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                            
        
        dp_g = gradients(p, x)[0]                                      
        p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                            

       # dp_g = gradients(p, x)[0]                                     
       # p_t, p_x = dp_g[:, :1], dp_g[:, 1:]                           
        
        dU2_g = gradients(U2, x)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = gradients(U3, x)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = gradients(F2, x)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = gradients(F3, x)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        d = (1/(0.2*(abs(u_x)-(u_x) )+1))
     
        f = ((d*(rho_t + U2_x - s1))**2).mean() + \
            ((d*(U2_t  + F2_x - s2))**2).mean() + \
            ((d*(U3_t  + F3_x - s3))**2).mean() #+\
            #((rho_t).mean())**2 +\
            #((U3_t).mean())**2
    
        return f

    def loss_ic(self, x, rho, u, p):
        y = self.forward(x)                                                      
        rho_nn, p_nn,u_nn = y[:, 0], y[:, 1], y[:, 2]  
        #rho_nn, p_nn,u_nn = y[0].squeeze() , y[1].squeeze() , y[2].squeeze()  

        loss_ics = ((u_nn - u) ** 2).mean() + \
               ((rho_nn- rho) ** 2).mean()  + \
               ((p_nn - p) ** 2).mean()

        return loss_ics
  
    def loss_rh(self, x,x_l):
        y = self.forward(x)                                    
        y_l = self.forward(x_l)                                    
        rho, p,u = y[:, 0], y[:, 1], y[:, 2]          
        rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]          
        #rho, p,u = y[0], y[1], y[2]
        #rhol, pl,ul = y_l[0], y_l[1], y_l[2]

        eta =  torch.clamp(abs(p-pl)-0.1,min=0)*torch.clamp(abs(u-ul)-0.1,min=0)

            
        loss_rh = ((rho*rhol*(u-ul)**2 -(pl-p)*(rhol - rho))**2*eta).mean()+\
                   (((rho*pl/0.4-rhol*p/0.4) - 0.5*(pl+p)*(rhol-rho))**2*eta).mean()#+\

        return loss_rh

    # Loss function for conservation
    def loss_con(self, x_en,x_in,crhoL,cuL,cpL,crhoR,cuR,cpR,t):
        y_en = self.forward(x_en)                                       
        y_in = self.forward(x_in)                                       
        rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         
        #rhoen, pen,uen = y_en[0], y_en[1], y_en[2]
        #rhoin, pin,uin = y_in[0], y_in[1], y_in[2]
        U3en = 0.5*rhoen*uen**2 + pen/0.4
        U3in = 0.5*rhoin*uin**2 + pin/0.4
        gamma = 0.4
        cU3L = 0.5*crhoL*cuL**2 + cpL/0.4 
        cU3R = 0.5*crhoR*cuR**2 + cpR/0.4 
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - t*(crhoL*cuL-crhoR*cuR))**2+ \
            ((-U3en+ U3in).mean() + t*(cU3L*cuL - cU3R*cuR) + (cpL*cuL - cpR*cuR)*t )**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(cpL-cpR)*t +(crhoL*cuL*cuL-crhoR*cuR*cuR)*t)**2
        return loss_en  
