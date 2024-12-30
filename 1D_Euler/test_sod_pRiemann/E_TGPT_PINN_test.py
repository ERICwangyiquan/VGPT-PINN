import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
#torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPT(nn.Module):
    def __init__(self, ini, layers, P, c_initial, Nx_train, Nt_train,gamma,Tf):
        super().__init__()
        self.layers = layers
        self.rhoL = ini[0]
        self.pL = ini[1]
        self.uL = ini[2]
        self.rhoR = ini[3]
        self.pR = ini[4]
        self.uR = ini[5]
        self.endt = Tf
        self.gamma = gamma
        self.Nx = Nx_train
        self.Nt = Nt_train
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], 1,bias=False)]+[nn.Linear(layers[1], 1,bias=False)]+[nn.Linear(layers[1], 1,bias=False)])
        self.activation = P
        
        for i in range(self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            self.linears[i].bias.data=torch.zeros(self.layers[0])

        self.linears[-3].weight.data =  c_initial
        self.linears[-2].weight.data =  c_initial
        self.linears[-1].weight.data =  c_initial
        self.linears[-3].weight.requires_grad = False
        self.linears[-2].weight.requires_grad = False
        self.linears[-1].weight.requires_grad = False

        
    def forward(self, x_data):
        test_data    = x_data
        u_rho = torch.Tensor().to(device)
        u_p = torch.Tensor().to(device)
        u_u = torch.Tensor().to(device)
        for i in range(0, self.layers[-2]):
            shift_data  = self.linears[i](test_data)
            u_data = self.activation[i](shift_data)
            u_rho = torch.cat((u_rho, u_data[:,[0]]), 1)
            u_p = torch.cat((u_p, u_data[:,[1]]), 1)
            u_u = torch.cat((u_u, u_data[:,[2]]), 1)
        output_rho = self.linears[-3](u_rho)
        output_p = self.linears[-2](u_p)
        output_u = self.linears[-1](u_u)
        return output_rho,output_p,output_u

    def WE_lossR(self, xt_residual, f_hat):
        """Residual loss function"""
        x = xt_residual.requires_grad_()
        y = self.forward(x)
        rho,p,u = y[0], y[1], y[2]
        
        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/(self.gamma-1)
        
        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)                                                 

        # Gradients and partial derivatives
        drho_g = autograd.grad(rho, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]                              
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g =autograd.grad(u, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]                                    
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                                                  
        
        dU2_g = autograd.grad(U2, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = autograd.grad(U3, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = autograd.grad(F2, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = autograd.grad(F3, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        d = 0.2*(abs(u_x)-u_x)  + 1 
        
        WE_loss = (((rho_t + U2_x)/d)**2).mean() + (((U2_t  + F2_x)/d)**2).mean() + (((U3_t  + F3_x)/d)**2).mean()
        return WE_loss
    
    def lossRH(self, x, x_l):
        y = self.forward(x)                                    
        y_l = self.forward(x_l)                                    
        #rho, p,u = y[:, 0], y[:, 1], y[:, 2]          
        #rhol, pl,ul = y_l[:, 0], y_l[:, 1], y_l[:, 2]          
        rho, p,u = y[0], y[1], y[2]
        rhol, pl,ul = y_l[0], y_l[1], y_l[2]

        eta =  torch.clamp(abs(p-pl)-0.1,min=0)*torch.clamp(abs(u-ul)-0.1,min=0)

        loss_rh = ((rho*rhol*(u-ul)**2 -(pl-p)*(rhol - rho))**2*eta).mean()+(((rho*pl/(self.gamma-1)-rhol*p/(self.gamma-1)) - 0.5*(pl+p)*(rhol-rho))**2*eta).mean()

        return loss_rh
    
    def lossCon(self, x_en,x_in):
        y_en = self.forward(x_en)                                       
        y_in = self.forward(x_in)                                       
        #rhoen, pen,uen = y_en[:, 0], y_en[:, 1], y_en[:, 2]         
        #rhoin, pin,uin = y_in[:, 0], y_in[:, 1], y_in[:, 2]         
        rhoen, pen,uen = y_en[0], y_en[1], y_en[2]
        rhoin, pin,uin = y_in[0], y_in[1], y_in[2]

        U3en = 0.5*rhoen*uen**2 + pen/(self.gamma-1)
        U3in = 0.5*rhoin*uin**2 + pin/(self.gamma-1)
        cU3L = 0.5*self.rhoL*self.uL**2 + self.pL/(self.gamma-1) 
        cU3R = 0.5*self.rhoR*self.uR**2 + self.pR/(self.gamma-1)
        # Loss function for the initial condition
        loss_en = ((rhoen - rhoin).mean() - self.endt*(self.rhoL*self.uL-self.rhoR*self.uR))**2+ \
            ((-U3en+ U3in).mean() + self.endt*(cU3L*self.uL - cU3R*self.uR) + (self.pL*self.uL - self.pR*self.uR)*self.endt)**2 +\
            ((-rhoen*uen + rhoin*uin).mean()+(self.pL-self.pR)*self.endt +(self.rhoL*self.uL*self.uL-self.rhoR*self.uR*self.uR)*self.endt)**2
        return loss_en   
    
    def lossICBC(self, ICBC_xt, ICBC_u):
        """Initial and both boundary condition loss function"""
        x=ICBC_xt.requires_grad_().to(device)
        y_ic = self.forward(x)
        rho_ic_nn, p_ic_nn,u_ic_nn = y_ic[0], y_ic[1], y_ic[2]   

        loss_ICBC = self.loss_function(rho_ic_nn.to(device), ICBC_u[0].to(device))+self.loss_function(p_ic_nn.to(device), ICBC_u[1].to(device))+self.loss_function(u_ic_nn.to(device), ICBC_u[2].to(device))
        return loss_ICBC

    def loss(self, xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test,xt_en, xt_RH, xt_RHL):
        """Total loss function"""
        WE_loss_R   = self.WE_lossR(xt_resid, f_hat)
        loss_IC = self.lossICBC(IC_xt, IC_u)
        loss_BC = self.lossICBC(BC_xt, BC_u)
        loss_RH = self.lossRH(xt_RH, xt_RHL)
        loss_con = self.lossCon(xt_en,IC_xt)
        loss = WE_loss_R + 10*(loss_IC+loss_BC) +100*loss_RH+100*loss_con
        return loss

    def lossR(self, xt_residual, f_hat):
        """Residual loss function"""
        x = xt_residual.requires_grad_()
        y = self.forward(x)
        rho,p,u = y[0], y[1], y[2]
        
        U2 = rho*u
        U3 = 0.5*rho*u**2 + p/(self.gamma-1)
        
        #F1 = U2
        F2 = rho*u**2+p
        F3 = u*(U3 + p)                                                 

        # Gradients and partial derivatives
        drho_g = autograd.grad(rho, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]                              
        rho_t, rho_x = drho_g[:, :1], drho_g[:, 1:]             


        du_g =autograd.grad(u, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]                                    
        u_t, u_x = du_g[:, :1], du_g[:, 1:]                                                  
        
        dU2_g = autograd.grad(U2, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        U2_t,U2_x = dU2_g[:,:1], dU2_g[:,1:]
        dU3_g = autograd.grad(U3, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        U3_t,U3_x = dU3_g[:,:1], dU3_g[:,1:]
        dF2_g = autograd.grad(F2, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        F2_t,F2_x = dF2_g[:,:1], dF2_g[:,1:]
        dF3_g = autograd.grad(F3, x, torch.ones(x.shape[0], 1).to(device),create_graph=True)[0]
        F3_t,F3_x = dF3_g[:,:1], dF3_g[:,1:]

        
        loss = (((rho_t + U2_x))**2).mean() + (((U2_t  + F2_x))**2).mean() + (((U3_t  + F3_x))**2).mean()
        return loss

    def loss_align(self, xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test,xt_en, xt_RH, xt_RHL):
        """Total loss function"""
        loss_R   = self.lossR(xt_resid, f_hat)
        loss_IC = self.lossICBC(IC_xt, IC_u)
        loss_BC = self.lossICBC(BC_xt, BC_u)

        loss = loss_R + loss_IC +loss_BC
        return loss
