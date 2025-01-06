import torch
torch.set_default_dtype(torch.float)
from E_Plotting import E_plot_2d

def gpt_train(VGPT_PINN, nu, x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train,x_test,ux_test,x_grid,y_grid,rx,ry,epochs_gpt, lr_gpt, tol_gpt):

    def closure():
        optimizer.zero_grad()
        loss_values=VGPT_PINN.loss(x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train)
        loss_values.backward()
        return loss_values
    
    optimizer = torch.optim.Adam(VGPT_PINN.parameters(), lr=lr_gpt)
    loss_values = optimizer.step(closure)
    losses=[loss_values.item()]
    ep=[0]
    print(f'{round(nu,3)} stopped at epoch: 0 | gpt_loss: {loss_values.item()}')

    lr_gpt = 0.0001
    optimizer = torch.optim.LBFGS(VGPT_PINN.parameters(),lr=lr_gpt,max_iter=20)
    for i in range(1, epochs_gpt+1):
        if (loss_values< tol_gpt): 
            losses.append(loss_values.item())
            # loss_R.append(loss_values[1].item())
            # loss_R0.append(loss_values[4].item())
            # loss_IC.append(loss_values[2].item())
            # loss_BC.append(loss_values[3].item())
            ep.append(i)
            print(f"layer1:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[0].bias.data} and layer3:{VGPT_PINN.linears[-1].weight.data}")
            print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values.item()} (VGPT_PINN Stopping Criteria Met)\n')
            break
            
        else:
            loss_values = optimizer.step(closure)
    

        if (i % 10 == 0) or (i == epochs_gpt):
            losses.append(loss_values.item())
            # loss_R.append(loss_values[1].item())
            # loss_R0.append(loss_values[4].item())
            # loss_IC.append(loss_values[2].item())
            # loss_BC.append(loss_values[3].item())
            ep.append(i)
            if (i % 50 == 0) or (i == epochs_gpt):
                print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values.item()}')
                print(f"layer1:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[0].bias.data} and layer3:{VGPT_PINN.linears[-1].weight.data}")
                if (i % 200 == 0):
                    u = VGPT_PINN(ux_test).detach().cpu().numpy()
                    E_plot_2d(x_test,x_grid,y_grid, u[:,0], 200, rx,ry,nu, title=fr"Density")
            #if (loss_values[0].item()<1e-5):
              #  ind = 0
             #   optimizer = torch.optim.LBFGS(GPT_PINN.parameters(), lr=lr_gpt)
            #    lr_gpt=0.1*lr_gpt
                    #   else:
                    #      lr_pinn=lr_gpt/0.7    
        if (i == epochs_gpt):
            print("VGPT-PINN Training Completed\n")         
    return loss_values, ep, losses