import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from B_data import initial_u,exact_u, initial_u2

from torch.autograd import Function

class NN2(nn.Module):
    def __init__(self, nu, layers):
        super().__init__()

        self.layers = layers
        self.nu = nu
        self.N = 100
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
            #if (i==0):
             #   padding_tensor = torch.zeros(a.shape[0],self.layers[1]-self.layers[0]).to(device)
              #  a = torch.cat((a, padding_tensor), dim=1)
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
    
    def lossRH(self, xt_residual,f_hat):
        """Residual loss function"""
        g = xt_residual.clone().requires_grad_()
        N_points = int(torch.sqrt(torch.tensor(g.shape[0])).item())
        u = self.forward(g)
        # u_plus = torch.roll(u, shifts=1)
        # u_plus[0]=u_plus[1]
        # u_minus = torch.roll(u, shifts=-1)
        # u_minus[-1]=u_minus[-2]
        # u_RH=(u_plus + u_minus) / 2
        # u_RH = u.clone()
        # for i in range(1, self.N-1):
        #     for j in range(self.N):
        #         upper = u[i-1, j]
        #         lower = u[i+1, j]
        #         u_RH[i*self.N+j] = (upper + lower) / 2
        u_up = u.clone().reshape(N_points,N_points)
        for col in range(N_points):
            u_up[col,:] = torch.roll(u_up[col,:], shifts=-1)
        u_up[:,-1] = u_up[:,-2].clone()
        u_down =u.clone().reshape(N_points,N_points)
        for col in range(N_points):
            u_down[col,:] = torch.roll(u_down[col,:], shifts=1)
        u_down[:,0] = u_down[:,1].clone()
        u_RH = ((u_up + u_down) / 2).reshape(u.shape[0], 1)
        u_hat_H=initial_u(self.nu,g[:,:1]-u_RH*g[:,1:]).to(device)
        u_hat = initial_u(self.nu,g[:,:1]-u*g[:,1:]).to(device)
        loss = self.wi*self.loss_function(u,u_hat.detach())+(0.1/self.wi)*self.loss_function(u,u_hat_H.detach())
        return loss

    def lossR(self,xt_data,f_hat,P_Iu):
        """Residual loss function"""
        g    = xt_data.clone().requires_grad_()
        u    = self.forward(g)
        #u = P(torch.cat((g[:,:1]-self.forward(g)*(g[:,1:]-1),torch.ones(g.shape[0],1).to(device)),1)).to(device)
        #u_plus = torch.roll(u, shifts=1)
        #u_plus[0]=u_plus[1]
        #u_minus = torch.roll(u, shifts=-1)
        #u_minus[-1]=u_minus[-2]
        #u_RH=(u_plus + u_minus) / 2
        #u_hat = MyFunction.apply(g[:,:1]-u*g[:,1:],torch.tensor(self.nu)).to(device)
        #u_hat = MyFunction.apply(g[:,:1]-u_RH*g[:,1:],torch.tensor(self.nu))
        #u_hat = MyFunction.apply(g[:,:1]-self.f_hat*g[:,1:],torch.tensor(self.nu))
        #u_hat=initial_u(self.nu,g[:,:1]-u*g[:,1:]).to(device)
        
        #u_hat = initial_u2(self.nu,g[:,:1]-f_hat*(g[:,1:]-1)).to(device)
        #u_hat = initial_u2(self.nu,g[:,:1]-u*(g[:,1:]-1)).to(device)
        #u_hat = P_Iu(torch.cat((g[:,:1]-f_hat*(g[:,1:]-1),torch.ones(g.shape[0],1).to(device)),1)).to(device)
        u_hat = P_Iu(torch.cat((g[:,:1]-u*(g[:,1:]-2/self.nu[1]),2/self.nu[1]*torch.ones(g.shape[0],1).to(device)),1)).to(device)

        #u_xt = autograd.grad(u, g, torch.ones(g.shape[0], 1).to(device),create_graph=True)[0]
        #u_x, u_t  = u_xt[:,:1],u_xt[:,1:]
        #f    = torch.add(u_t,torch.mul(u, u_x))
        #d = 0.1*(abs(u_x))+1

        return self.loss_function(u,u_hat.detach())
        #return self.loss_function(f/d,f_hat/d)
    
    def lossIC(self, IC_xt, PINN_u):
        """First initial loss function"""
        x = IC_xt.clone().requires_grad_()
        return self.loss_function(self.forward(x), PINN_u)


    def lossBC(self,BC1, BC2):
        """Both boundary condition loss function"""
        #B1 = self.BC_bottom.clone().requires_grad_()
        ##B2 = self.BC_top.clone().requires_grad_()
        #D_BC = self.nu*torch.ones(B1.shape[0],1).to(device)
        ##return self.loss_function(self.forward(B1), self.forward(B2))
        #return self.loss_function(self.forward(B1), D_BC)
        # BC_b1 = BC1[:,0:2].clone().requires_grad_()
        # BC_b2 = BC1[:,2:4].clone().requires_grad_()
        # loss_bc_bottom = self.loss_function(self.forward(BC_b1), self.forward(BC_b2))
        # BC_t1 = BC2[:,0:2].clone().requires_grad_()
        # BC_t2 = BC2[:,2:4].clone().requires_grad_()
        # loss_bc_top = self.loss_function(self.forward(BC_t1), self.forward(BC_t2))
        # loss = loss_bc_bottom+loss_bc_top
        #g1 = BC1.clone().requires_grad_()
        #u_BC1 = self.forward(g1)
        #du_BC1 = autograd.grad(u_BC1, g1, torch.ones(g1.shape[0], 1).to(device), create_graph=True)[0]
        #dx_BC1 = du_BC1[:,[0]]
        #loss_BC1 = self.loss_function(dx_BC1, torch.zeros(BC1.shape[0],1).to(device))
        #g2 = BC2.clone().requires_grad_()
        #u_BC2 = self.forward(g2)
        #du_BC2 = autograd.grad(u_BC2, g2, torch.ones(g2.shape[0], 1).to(device), create_graph=True)[0]
        #dx_BC2 = du_BC2[:,[0]]
        #loss_BC2 = self.loss_function(dx_BC2, torch.zeros(BC2.shape[0],1).to(device))
        #loss = loss_BC1 + loss_BC2+self.loss_function(dx_BC1,dx_BC2)       
        g1 = BC1.clone().requires_grad_()
        u_BC1 = self.forward(g1)
        #du_BC1 = autograd.grad(u_BC1, g1, torch.ones(g1.shape[0], 1).to(device), create_graph=True)[0]
        #dx_BC1 = du_BC1[:,[0]]
        loss_BC1 = self.loss_function(u_BC1,self.nu[1]*torch.ones(BC1.shape[0],1).to(device))
        g2 = BC2.clone().requires_grad_()
        u_BC2 = self.forward(g2)
        #du_BC2 = autograd.grad(u_BC2, g2, torch.ones(g2.shape[0], 1).to(device), create_graph=True)[0]
        #dx_BC2 = du_BC2[:,[0]]
        loss_BC2 = self.loss_function(u_BC2, torch.zeros(BC2.shape[0],1).to(device))
        loss = loss_BC1 + loss_BC2
        return loss
 
    def loss(self, xt_resid, xt_test,IC_xt, IC_u, BC1, BC2,f_hat,P):
        """Total Loss Function"""
        loss_R   = self.lossR(xt_resid,f_hat,P)
        loss_IC = self.lossIC(IC_xt, IC_u)
        loss_BC  = self.lossBC(BC1, BC2)
        #loss_RH = self.lossRH(xt_test)
        loss = loss_R+loss_BC+loss_IC
        return loss, loss_R,loss_IC,loss_BC