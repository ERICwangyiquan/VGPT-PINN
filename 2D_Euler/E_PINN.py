import torch
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,grad_outputs=torch.ones_like(outputs), create_graph=True)


class NN(nn.Module):    
    def __init__(self, nu,layers, gamma):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.gamma = gamma
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
        a = self.linears[-1](a)
        return a
    
    def bd_B(self,x,sin,cos):
        yb = self.forward(x)
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
    def bd_OY(self,x):
        y = self.forward(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_y)/lam)**2).mean() +\
            ((( v_y)/lam)**2).mean() +\
            ((( p_y)/lam)**2).mean() +\
            ((( rho_y)/lam)**2).mean()
        return f
    
    def bd_OX(self,x):
        y = self.forward(x)
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        drho_g = gradients(rho, x)[0]                                  # Gradient [rho_t, rho_x]
        rho_x,rho_y = drho_g[:, :1], drho_g[:, 1:2]                    # Partial derivatives rho_t, rho_x
        du_g = gradients(u, x)[0]                                      # Gradient [u_t, u_x]
        u_x, u_y = du_g[:, :1], du_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dv_g = gradients(v, x)[0]                                      # Gradient [u_t, u_x]
        v_x, v_y = dv_g[:, :1], dv_g[:, 1:2]                            # Partial derivatives u_t, u_x
        dp_g = gradients(p, x)[0]                                      # Gradient [p_t, p_x]
        p_x, p_y = dp_g[:, :1], dp_g[:, 1:2]                            # Partial derivatives p_t, p_x
        
        deltau = u_x + v_y
        lam = 0.1*(abs(deltau) - deltau) + 1
        
        f = ((( u_x)/lam)**2).mean() +\
            ((( v_x)/lam)**2).mean() +\
            ((( p_x)/lam)**2).mean() +\
            ((( rho_x)/lam)**2).mean()
        return f

    # Loss function for PDE
    def loss_pde(self, x):
        
        y = self.forward(x)
        gamma = 1.4                                                   
        epsilon = 1e-5
        rho,p,u,v = y[:, 0:1], y[:, 1:2], y[:, 2:3],y[:,3:]
        
        rhoE = p/(gamma - 1) +0.5*rho*(u**2+v**2)
        
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
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics

    def loss_bc(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.forward(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((u_ic_nn - u_ic) ** 2).mean() + \
               ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() +\
               ((v_ic_nn - v_ic) ** 2).mean()

        return loss_ics
    def loss_bc1(self, x_ic, rho_ic, u_ic, v_ic,p_ic):
        U_ic = self.forward(x_ic)                                                      # Initial condition
        rho_ic_nn, p_ic_nn,u_ic_nn,v_ic_nn = U_ic[:, 0], U_ic[:, 1], U_ic[:, 2],U_ic[:,3]            # rho, u, p - initial condition

        # Loss function for the initial condition
        loss_ics = ((rho_ic_nn- rho_ic) ** 2).mean()  + \
               ((p_ic_nn - p_ic) ** 2).mean() 

        return loss_ics

    def loss_rh(self, x,x_l):
        y = self.net(x)                                    
        y_l = self.net(x_l)                                    
        rho, p,u,v = y[:, 0], y[:, 1], y[:, 2],y[:,3]          
        rhol, pl,ul,vl = y_l[:, 0], y_l[:, 1], y_l[:, 2],y_l[:,3 ]          

      #  du_g = gradients(u, x)[-1]                                      
      #  u_t, u_x = du_g[:, -1], du_g[:, 1]                            
      #  d = 0/(0.1*(abs(u_x)-u_x)  + 1)
        #eta =  torch.clamp(d-1.1,max=0)*torch.clamp(abs(pr-pl)-0.1,min=0)#*torch.clamp(abs(ur-ul)-0.1,min=0)
        eta =  torch.clamp(abs(p-pl)-0.2,min=0)*torch.clamp((u-ul)**2+(v-vl)**2-0.04,min=0)
            
        loss_rh = ((rho*rhol*((u-ul)**2+ (v-vl)**2)-(pl-p)*(rhol - rho))**2*eta).mean()+\
                   (((rho*pl/0.4-rhol*p/0.4) - 0.5*(pl+p)*(rhol-rho))**2*eta).mean()#+\
        return loss_rh


    def loss(self, x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train):
        """Total loss function"""
        loss_pde = self.loss_pde(x_int_train)                                   
        loss_ic = self.loss_ic(x_ic_train, rho_ic_train,u_ic_train,v_ic_train,p_ic_train)   
        loss_bdL = self.loss_bc(x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train)   
        # loss_bdR = self.loss_bc1(x_bcR_train,rho_bcR_train,u_bcR_train,v_bcR_train,p_bcR_train)   
        loss_bdI = self.bd_B(x_bcI_train, sin_bcI_train,cos_bcI_train)  

        loss_ib = loss_ic  +  loss_bdI +loss_bdL
        loss = loss_pde + 10*loss_ib

        return loss