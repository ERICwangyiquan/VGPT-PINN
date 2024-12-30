import torch
torch.set_default_dtype(torch.float)    

def pinn_train(PINN, nu, x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train, epochs_pinn, lr_pinn, tol):

    def closure():
        optimizer.zero_grad()
        loss = PINN.loss(x_int_train,x_ic_train,rho_ic_train,u_ic_train,v_ic_train,p_ic_train,x_bcL_train, rho_bcL_train,u_bcL_train,v_bcL_train,p_bcL_train,x_bcI_train, sin_bcI_train,cos_bcI_train)
        loss.backward()
        return loss

    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    loss_values = optimizer.step(closure)

    lr_pinn = 0.0001
    optimizer = torch.optim.LBFGS(PINN.parameters(),lr=1,max_iter=20)
    print(f"Epoch: 0 | Loss: {loss_values.item()}")
    for i in range(1, epochs_pinn):
        if (loss_values.item() < tol):
            print(f'Epoch: {i} | Loss: {loss_values.item():.6f} (Stopping Criteria Met)')
            break
        else :
            loss_values = optimizer.step(closure)
            if (i % 100 == 0):
                print(f'Epoch: {i} | Loss: {loss_values.item():.6f}')
                if (i == epochs_pinn):
                    print("PINN Training Completed\n")
    return loss_values