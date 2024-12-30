import matplotlib.pyplot as plt
from numpy import ubyte
import torch
from torch import cos, sin
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from B_data import initial_u,exact_u

class NN(nn.Module):    
    def __init__(self, nu, layers,dt):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.dt = dt
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])


        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = nn.Tanh()


    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
            #if(i==0):
            #   a = a + z
            #   a0 = a
            #if(i>0):
            #   a=a+a0
            #   a0=a
        #a = a + a0
        a = self.linears[-1](a)
        return a

   
    def lossIMR(self, xt_residual):
        """Residual loss function"""
        g = xt_residual.clone().requires_grad_()
        u = self.forward(g)
        u_hat = initial_u(self.nu,g[:,:1]-u*g[:,1:]).to(device)
        #u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        #u_x = u_xt[:,[0]] 
        #d = (0.1*(torch.abs(u_x))+1)
        loss = self.loss_function(u,u_hat.detach())
        #loss = self.loss_function(u/d,(u_hat/d).detach())
        return loss

    def lossWER(self, xt_residual,f_hat):
        """Residual loss function"""
        g = xt_residual.clone().requires_grad_()
        u = self.forward(g)
        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        #u_xx_tt = autograd.grad(u_xt, g, torch.ones(g.shape).to(device), create_graph=True)[0]
        #u_xx = u_xx_tt[:,[0]] 

        u_x = u_xt[:,[0]]        
        u_t = u_xt[:,[1]]                
        f1 = torch.mul(u, u_x)
        #f2 = torch.mul(-0.005,u_xx)
        f = torch.add(u_t, f1)
        d = (0.1*(torch.abs(u_x)-u_x)+1)
        
        loss_R = self.loss_function(f/d,f_hat/d)
        return loss_R
    
    def lossRH(self, xt_RH, xt_RHL,xt_RHR,xt_RHt,xt_RHtL):
        y = self.forward(xt_RH)                                    
        y_l = self.forward(xt_RHL)
        y_r = self.forward(xt_RHR)  
        #eta =  torch.clamp(abs(y-y_l)-0.2,min=0)
        eta = torch.where(abs(y-y_l)>0.1,1.0,0.0)
        y_dt = self.forward(xt_RHt)
        y_dl = self.forward(xt_RHtL)
        #eta_dt =  torch.clamp(abs(y_dt-y_dl)-0.2,min=0)
        eta_dt =torch.where(abs(y_dt-y_dl)>0.1,1.0,0.0)
        #loss_rh = (((self.nu/2*self.xt_RH[:,1]-(y-y_l))*eta)**2).mean()
        #loss_rh = ((y*(self.nu-y_l)*eta)**2).mean()
        #loss_rh = ((((y+y_l)/2*self.xt_RH[:,1]-(y-y_l))*eta)**2).mean()
        #loss_rh = (((y-y_i-y*self.xt_RH[:,1:])*eta)**2).mean()
        s = (y_r+y_l)/2
        x = xt_RH[:,:1]+s*self.dt
        loss_rh = self.loss_function(x*eta,xt_RHt[:,:1]*eta_dt)
        return loss_rh
    
    def lossCon(self, x_en,x_in):
        y_en = self.forward(x_en)                                       
        y_in = self.forward(x_in)    

        loss_con = ((y_en - y_in).mean())**2
        return loss_con
    
    def lossIC(self, IC_xt, IC_u):
        """Initial and both boundary condition loss function"""
        g = IC_xt.clone().requires_grad_()
        u_IC = self.forward(g)
        #u_xt = autograd.grad(u_IC, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        #u_x = u_xt[:,[0]]
        #d = 0.1*(torch.abs(u_x))+1
        loss_IC = self.loss_function(u_IC, IC_u)
        return loss_IC

    def lossBC(self, BC1, BC2):
        g1 = BC1.clone().requires_grad_()
        u_BC1 = self.forward(g1)
        g2 = BC2.clone().requires_grad_()
        u_BC2 = self.forward(g2)
        loss_BC1 = self.loss_function(u_BC1,torch.zeros(BC1.shape[0],1).to(device))
        loss_BC2 = self.loss_function(u_BC2,self.nu*torch.ones(BC1.shape[0],1).to(device))
        loss = loss_BC1+loss_BC2
        return loss

    def loss(self,xt_resid, xt_test,IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat):
        """Total loss function"""
        loss_R   = self.lossIMR(xt_resid)
        #loss_R   = self.lossWER(xt_resid,f_hat)
        loss_IC = self.lossIC(IC_xt, IC_u)
        loss_BC  = self.lossBC(BC1, BC2)
        loss_RH = self.lossRH(xt_resid, xt_RHL,xt_RHR,xt_RHt,xt_RHtL)
        loss_con = self.lossCon(BC1,BC2)
        loss = loss_R+10*(loss_IC +loss_BC)+10*loss_RH
        return loss, loss_R,loss_IC,loss_BC,loss_RH,loss_con