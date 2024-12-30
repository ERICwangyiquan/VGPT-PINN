import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)


class GPT(nn.Module):
    def __init__(self, rd, layers, P, c_initial, gamma):
        super().__init__()
        self.layers = layers
        self.rd = rd
        self.gamma = gamma
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[0], layers[0]) for i in range(self.layers[1])]+[nn.Linear(layers[1], 1,bias=False)]+[nn.Linear(layers[1], 1,bias=False)]+[nn.Linear(layers[1], 1,bias=False)]+[nn.Linear(layers[1], 1,bias=False)])
        #self.linears = nn.ModuleList([nn.Linear(layers[0], layers[1]),nn.Linear(layers[1], layers[2]),nn.Linear(layers[2], layers[3],bias=False)])
        self.activation = P
        
        for i in range(0,self.layers[1]):
            self.linears[i].weight.data = torch.eye(self.layers[0])
            self.linears[i].bias.data = torch.zeros(self.layers[0])

        self.linears[-4].weight.data =  c_initial
        self.linears[-3].weight.data =  c_initial
        self.linears[-2].weight.data =  c_initial
        self.linears[-1].weight.data =  c_initial
    

    def forward(self, x_data):
        test_data    = x_data.float()
        u_rho = torch.Tensor().to(device)
        u_p = torch.Tensor().to(device)
        u_u = torch.Tensor().to(device)
        u_v = torch.Tensor().to(device)
        for i in range(0, self.layers[-2]):
            shift_data  = self.linears[i](test_data)
            u_data = self.activation[i](shift_data)
            u_rho = torch.cat((u_rho, u_data[:,[0]]), 1)
            u_p = torch.cat((u_p, u_data[:,[1]]), 1)
            u_u = torch.cat((u_u, u_data[:,[2]]), 1)
            u_v = torch.cat((u_v, u_data[:,[3]]), 1)
        output_rho = self.linears[-4](u_rho)
        output_p = self.linears[-3](u_p)
        output_u = self.linears[-2](u_u)
        output_v = self.linears[-1](u_v)
        return torch.hstack((output_rho,output_p,output_u,output_v))

    def bd_B(self,x,sin,cos):
        yb = self.forward(x)
        #rhob,pb,ub,vb = yb[0], yb[1], yb[2],yb[3]
        rhob,pb,ub,vb = yb[:, 0:1], yb[:, 1:2], yb[:, 2:3],yb[:,3:]
        drhob_g = gradients(rhob, x)[0]                                      # Gradient [u_t, u_x]
        rhob_x, rhob_y = drhob_g[:, 1:2], drhob_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dub_g = gradients(ub, x)[0]                                      # Gradient [u_t, u_x]
        ub_x, ub_y = dub_g[:, 1:2], dub_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dvb_g = gradients(vb, x)[0]                                      # Gradient [u_t, u_x]
        vb_x, vb_y = dvb_g[:, 1:2], dvb_g[:, 2:3]                            # Partial derivatives u_t, u_x
        dpb_g = gradients(pb, x)[0]                                      # Gradient [p_t, p_x]
        pb_x, pb_y = dpb_g[:, 1:2], dpb_g[:, 2:3]                            # Partial derivatives p_t, p_x
        
        deltau = ub_x + vb_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        #lam = (deltau) - deltau) + 1
        
        fb = (((ub*cos + vb*sin)/lam)**2).mean() +\
            (((pb_x*cos + pb_y*sin)/lam)**2).mean() +\
            (((rhob_x*cos + rhob_y*sin)/lam)**2).mean()
        return fb

    # Loss function for PDE
    def loss_pde(self, x):
        
        y = self.forward(x)                                                  
        epsilon = 1e-5
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        #rho,p,u,v = y[0], y[1], y[2],y[3]
        
        rhoE = p/(self.gamma - 1) +0.5*rho*(u**2+v**2)
        
        f1 = rho*u
        f2 = rho*u*u+p
        f3 = rho*u*v
        f4 = (rhoE+p)*u
        
        g1 = rho*v
        g2 = rho*v*u
        g3 = rho*v*v + p
        g4 = (rhoE+p)*v
        
        drho_g = gradients(rho,x)[0]
        U1_t = drho_g[:, :1]
        dU2_g = gradients(f1,x)[0]
        U2_t = dU2_g[:, :1]
        dU3_g = gradients(g1,x)[0]
        U3_t = dU3_g[:, :1]
        dU4_g = gradients(rhoE,x)[0]
        U4_t = dU4_g[:, :1]
        
        df1_g = gradients(f1, x)[0]                                  
        f1_x = df1_g[:, 1:2]
        df2_g = gradients(f2, x)[0]                                  
        f2_x = df2_g[:, 1:2]
        df3_g = gradients(f3, x)[0]                                  
        f3_x = df3_g[:, 1:2]
        df4_g = gradients(f4, x)[0]                                  
        f4_x = df4_g[:, 1:2]
        
        dg1_g = gradients(g1, x)[0]                                  
        g1_y = dg1_g[:, 2:3]
        dg2_g = gradients(g2, x)[0]                                  
        g2_y = dg2_g[:, 2:3]
        dg3_g = gradients(g3, x)[0]                                  
        g3_y = dg3_g[:, 2:3]
        dg4_g = gradients(g4, x)[0]                                  
        g4_y = dg4_g[:, 2:3]
        
        
        du_g = gradients(u, x)[0]                                
        u_x = du_g[:, 1:2]         
        dv_g = gradients(v, x)[0]                    
        v_y = dv_g[:, 2:3]         
        
        deltau = u_x + v_y
        nab = abs(deltau) - deltau
        
        lam = 0.1*nab + 1
        
        f = (((U1_t + f1_x+g1_y )/lam)**2).mean() +\
            (((U2_t + f2_x+g2_y )/lam)**2).mean() +\
            (((U3_t + f3_x+g3_y )/lam)**2).mean() +\
            (((U4_t + f4_x+g4_y )/lam)**2).mean()

        return f

    # Loss function for initial condition
    def loss_ic(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.forward(x_ic)                                                      # Initial condition
        #rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[0], U_ic[1], U_ic[2],U_ic[3]
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:,0:1], U_ic[:,1:2], U_ic[:,2:3],U_ic[:,3:]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic[:,None]) ** 2).mean() + \
               ((rho_ic_nn- rho_ic[:,None]) ** 2).mean()  + \
               ((p_ic_nn - p_ic[:,None]) ** 2).mean() +\
               ((v_ic_nn - v_ic[:,None]) ** 2).mean()

        return loss_ics

    def loss_bc(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.forward(x_ic)                                                      # Initial condition
        #rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[0], U_ic[1], U_ic[2],U_ic[3]
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:,0:1], U_ic[:,1:2], U_ic[:,2:3],U_ic[:,3:]
        
        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic[:,None]) ** 2).mean() + \
               ((rho_ic_nn- rho_ic[:,None]) ** 2).mean()  + \
               ((p_ic_nn - p_ic[:,None]) ** 2).mean() +\
               ((v_ic_nn - v_ic[:,None]) ** 2).mean()

        return loss_ics


    def loss(self, x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train):
        """Total loss function"""
        loss_pde = self.loss_pde(x_int_train)                                   
        loss_ic = self.loss_ic(x_ic_train, rho_ic_train,u_ic_train,v_ic_train,p_ic_train)   
        loss_bdL = self.loss_bc(x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train)   
        # loss_bdR = self.loss_bc1(x_bcR_train,rho_bcR_train,u_bcR_train,v_bcR_train,p_bcR_train)   
        loss_bdI = self.bd_B(x_bcI_train, sin_bcI_train,cos_bcI_train)  

        loss_ib = loss_ic  +  loss_bdI +loss_bdL
        loss = loss_pde + 10*loss_ib

        return loss,loss_pde,loss_ic,loss_bdL,loss_bdI