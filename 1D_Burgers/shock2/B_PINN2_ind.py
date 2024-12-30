import matplotlib.pyplot as plt
from numpy import ubyte
import torch
from torch import cos, sin
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from B_data import initial_u

class NN2(nn.Module):    
    def __init__(self, nu, layers,dt,cut_t=None):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.dt = dt
        self.cut_t = cut_t
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
    
    def lossIMR(self,xt_data,f_hat,P_Iu):
        """Residual loss function"""
        g    = xt_data.clone().requires_grad_()
        u    = self.forward(g)
        #range_ind = torch.logical_and(g[:,:1]-u*(g[:,1:]-2/self.nu[1]) >= -1.0, g[:,:1]-u*(g[:,1:]-2/self.nu[1]) < 1.5)
        #u_hat = P_Iu(torch.cat((g[:,:1]-u*(g[:,1:]-2/self.nu[1]),2/self.nu[1]*torch.ones(g.shape[0],1).to(device)),1)).to(device)
        range_ind = torch.logical_and(g[:,:1]-u*(g[:,1:]-self.cut_t) >= -1.0, g[:,:1]-u*(g[:,1:]-self.cut_t) < 1.5)
        u_hat = P_Iu(torch.cat((g[:,:1]-u*(g[:,1:]-self.cut_t),self.cut_t*torch.ones(g.shape[0],1).to(device)),1)).to(device)
        u_hat = initial_u(self.nu,xt_data[:,:1]-u_hat*xt_data[:,1:]).to(device)
        loss_range = self.loss_function(u[range_ind],u_hat[range_ind].detach())

        return loss_range


    def lossIC(self, IC_xt, PINN_u):
        """First initial loss function"""
        x = IC_xt.clone().requires_grad_()
        return self.loss_function(self.forward(x), PINN_u)


    def lossBC(self,BC1, BC2):
        """Both boundary condition loss function"""     
        g1 = BC1.clone().requires_grad_()
        u_BC1 = self.forward(g1)
        loss_BC1 = self.loss_function(u_BC1,self.nu[1]*torch.ones(BC1.shape[0],1).to(device))
        g2 = BC2.clone().requires_grad_()
        u_BC2 = self.forward(g2)
        loss_BC2 = self.loss_function(u_BC2, torch.zeros(BC2.shape[0],1).to(device))
        loss = loss_BC1 + loss_BC2
        return loss

    def loss(self,xt_resid, xt_test,IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat,P1):
        """Total loss function"""
        loss_R = self.lossIMR(xt_resid,f_hat,P1)
        #loss_R0   = self.lossWER(xt_resid,f_hat)
        loss_IC = self.lossIC(IC_xt, IC_u)
        loss_BC  = self.lossBC(BC1, BC2)
        loss_RH = self.lossRH(xt_resid, xt_RHL,xt_RHR,xt_RHt,xt_RHtL)
        loss_con = self.lossCon(BC1,BC2)
        loss = loss_R+10*(loss_IC +loss_BC)+10*loss_RH
        return loss, loss_R,loss_IC,loss_BC,loss_RH,loss_con