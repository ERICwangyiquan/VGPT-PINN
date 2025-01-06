import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from B_data import initial_u,exact_u

class GPT(nn.Module):
    def __init__(self, nu, layers, P, c_initial, resid_data, f_hat, u_exact, xt_test, IC_data,IC_u, BC_bottom, BC_top,xt_RHL,xt_RHR, xt_RH0,xt_RH0L,Nx_train, Nt_train,dt):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.Nx = Nx_train
        self.Nt = Nt_train
        self.dt = dt
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], layers[2])])
        self.activation = P
        
        self.resid_data       = resid_data
        self.IC_data          = IC_data
        self.BC_bottom        = BC_bottom
        self.BC_top           = BC_top
        self.IC_u             = IC_u
        self.f_hat            = f_hat
        self.u_exact          = u_exact
        self.xt_test          = xt_test
        self.xt_RH            = resid_data
        self.xt_RHL           = xt_RHL
        self.xt_RHR           = xt_RHR
        self.xt_RH0           = xt_RH0
        self.xt_RH0L          = xt_RH0L
        
        for i in range(self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            #self.linears[i].weight.data[1][1] = self.nu 
            self.linears[i].bias.data=torch.zeros(self.layers[0])

        self.linears[-1].weight.data =  c_initial
        #self.linears[-1].weight.data[0][0] =  torch.tensor(self.nu)
        self.linears[-1].bias.data=torch.zeros(self.layers[1])
        
    def forward(self, x_data):
        test_data    = x_data.float()
        u_data = torch.Tensor().to(device)
        for i in range(0, self.layers[-2]):
            shift_data  = self.linears[i](test_data)
            #xshift_data = (shift_data[:,:1]+1)% 2 - 1
            #tshift_data = shift_data[:,1:]
            #u_data = torch.cat((u_data, self.activation[i](torch.cat((xshift_data,tshift_data),1))), 1)
            u_data = torch.cat((u_data, self.activation[i](shift_data)), 1)
        #u_data       = self.activation(torch.cat((xshift_data.reshape(test_data[:,0].shape[0],1),torch.zeros(test_data[:,0].shape[0],1).to(device)),1))
        #u_data       = self.activation[0](xshift_data)
        final_output = self.linears[-1](u_data)
        return final_output

    def lossR(self,x):
        """Residual loss function"""
        g    = x.clone().requires_grad_()
        u    = self.forward(g)
        #u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        #u_x = u_xt[:,[0]] 
        #d = (0.1*(torch.abs(u_x))+1)
        #u_plus = torch.roll(u, shifts=1)
        #u_plus[0]=u_plus[1]
        #u_minus = torch.roll(u, shifts=-1)
        #u_minus[-1]=u_minus[-2]
        #u_RH=(u_plus + u_minus) / 2
        #u_hat_H=initial_u(self.nu,g[:,:1]-u_RH*g[:,1:]).to(device)
        
        u_hat=initial_u(self.nu,g[:,:1]-u*g[:,1:]).to(device)
        #loss = self.wi*self.loss_function(u,u_hat.detach())+(1/self.wi)*self.loss_function(u[self.ind],u_hat_H[self.ind].detach())
        #loss = self.wi*self.loss_function(u,u_hat.detach())+(1/self.wi)*self.loss_function(u,u_hat_H.detach())
        #loss = 0.5*self.loss_function(u,u_hat.detach())+0.5*self.loss_function(u,u_hat_H.detach())
        loss = self.loss_function(u,u_hat.detach())
        return loss
        #return self.loss_function(f/d,self.f_hat/d)

    def lossRH(self):
        """Residual loss function"""
        y = self.forward(self.xt_RH)                                    
        y_l = self.forward(self.xt_RHL)  
        y_r = self.forward(self.xt_RHR)
        eta = torch.where(abs(y-y_l)>0.1,1.0,0.0)
        y_dt = self.forward(self.xt_RH0)
        y_dl = self.forward(self.xt_RH0L)
        eta_dt =torch.where(abs(y_dt-y_dl)>0.1,1.0,0.0)
        #loss_rh = (((self.nu/2*self.xt_RH[:,1]-(y-y_l))*eta)**2).mean()
        #loss_rh = ((y*(self.nu-y_l)*eta)**2).mean()
        #loss_rh = ((((y+y_l)/2*self.xt_RH[:,1]-(y-y_l))*eta)**2).mean()
        #loss_rh = (((y-y_i-y*self.xt_RH[:,1:])*eta)**2).mean()
        s = (y_r+y_l)/2
        x = self.xt_RH[:,:1]+s*self.dt
        loss_rh = self.loss_function(x*eta,self.xt_RH0[:,:1]*eta_dt)
        return loss_rh

    def lossR0(self):
        """Residual loss function"""
        g = self.resid_data.clone().requires_grad_()
        u = self.forward(g)

        u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device), create_graph=True)[0]
        #u_xx_tt = autograd.grad(u_xt, g, torch.ones(g.shape).to(device), create_graph=True)[0]
        #u_xx = u_xx_tt[:,[0]] 

        u_x = u_xt[:,[0]]        
        u_t = u_xt[:,[1]]                
        f1 = torch.mul(u, u_x)
        #f2 = torch.mul(-0.005,u_xx)
        f = torch.add(u_t, f1)
        d = (0.1*(torch.abs(u_x))+1)
        
        loss_R = self.loss_function(f/d,self.f_hat/d)
        return loss_R
    
    def lossIC(self):
        """First initial loss function"""
        x = self.IC_data.clone().requires_grad_()
        return self.loss_function(self.forward(x), self.IC_u)


    def lossBC(self):
        g1 = self.BC_bottom.clone().requires_grad_()
        u_BC1 = self.forward(g1)
        g2 = self.BC_top.clone().requires_grad_()
        u_BC2 = self.forward(g2)
        loss = self.loss_function(u_BC1,u_BC2)
        return loss
    
    def lossMSE(self):
        """Mean Square Error loss function"""
        u = self.forward(self.xt_test)
        return self.loss_function(u, self.u_exact)

    def loss(self,xt):
        """Total Loss Function"""
        loss_R   = self.lossR(xt)
        loss_R0   = self.lossR0()
        loss_IC  = self.lossIC()
        loss_BC  = self.lossBC()
        loss_RH = self.lossRH()
        loss = loss_R + 10*(loss_IC+loss_BC) + 10*loss_RH 
        #loss = self.wi*loss_R+loss_IC +loss_BC+(1/self.wi)*loss_R0
        #loss = self.lossMSE()
        return loss, loss_R, loss_IC,loss_BC,loss_RH