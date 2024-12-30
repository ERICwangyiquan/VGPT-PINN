import torch
torch.set_default_dtype(torch.float)
from B_data import exact_u
import torch.utils.data as data

def gpt_train(GPT_PINN, nu, xt_resid,f_hat, Exact_y0, xt_test, IC_xt, IC_u, BC1, BC2, epochs_gpt, lr_gpt, tol_gpt):

    ind =1
    #Exact_y0 = exact_u(nu,xt_test)[:,None].clone()
    rMAE = [max(sum(abs(GPT_PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()]
    rRMSE = [torch.sqrt(sum((GPT_PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()]

    optimizer = torch.optim.Adam(GPT_PINN.parameters(), lr=lr_gpt)

    loss_values = GPT_PINN.loss(xt_resid)
    losses=[loss_values[0].item()]
    ep=[0]
    loss_R  =[loss_values[1].item()]
    loss_R0 =[loss_values[4].item()]
    loss_IC =[loss_values[2].item()]
    loss_BC =[loss_values[3].item()]
    print(f'{round(nu,3)} stopped at epoch: 0 | gpt_loss: {loss_values[0].item()},rMAE: {rMAE}, rRMSE:{rRMSE}, {loss_values[1].item()}, {loss_values[2].item()}, {loss_values[3].item()}, {loss_values[4].item()}')
    def closure():
        optimizer.zero_grad()
        loss_values=GPT_PINN.loss(xt_resid) 
        loss_values[0].backward()
        return loss_values

    for i in range(1, epochs_gpt+1):
        if (loss_values[0] < tol_gpt): 
            L1_loss = max(sum(abs(GPT_PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()
            L2_loss = torch.sqrt(sum((GPT_PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()
            rMAE.append(L1_loss)
            rRMSE.append(L2_loss)
            losses.append(loss_values[0].item())
            loss_R.append(loss_values[1].item())
            loss_R0.append(loss_values[4].item())
            loss_IC.append(loss_values[2].item())
            loss_BC.append(loss_values[3].item())
            ep.append(i)
            print(f"layer1:{GPT_PINN.linears[0].weight.data} and {GPT_PINN.linears[0].bias.data} and layer3:{GPT_PINN.linears[-1].weight.data}")
            print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values[0].item()} ,rMAE: {L1_loss}, rRMSE:{L2_loss}(GPT_PINN Stopping Criteria Met)\n')
            break
            
        else:
            if (loss_values[0].item()<10*tol_gpt):#1e-2for1
                GPT_PINN.wi.requires_grad = False
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, GPT_PINN.parameters()), lr=lr_gpt)
                #print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values[0].item()},rMAE: {L1_loss}, rRMSE:{L2_loss}, {loss_values[1].item()}, {loss_values[2].item()}, {loss_values[3].item()}, {loss_values[4].item()}')

        loss_values = optimizer.step(closure)

        if (i % 100 == 0) or (i == epochs_gpt):
            #print({GPT_PINN.wi.data.item()})
            L1_loss = max(sum(abs(GPT_PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()
            L2_loss = torch.sqrt(sum((GPT_PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()
            rMAE.append(L1_loss)
            rRMSE.append(L2_loss)
            losses.append(loss_values[0].item())
            loss_R.append(loss_values[1].item())
            loss_R0.append(loss_values[4].item())
            loss_IC.append(loss_values[2].item())
            loss_BC.append(loss_values[3].item())
            ep.append(i)
            if (i % 500 == 0) or (i == epochs_gpt):
                #print(GPT_PINN.wi)
                print(f'{round(nu,3)} stopped at epoch: {i} | gpt_loss: {loss_values[0].item()},rMAE: {L1_loss}, rRMSE:{L2_loss}, {loss_values[1].item()}, {loss_values[2].item()}, {loss_values[3].item()}, {loss_values[4].item()}')
                #print(f"layer1:{GPT_PINN.linears[0].weight.data} and {GPT_PINN.linears[0].bias.data} and layer3:{GPT_PINN.linears[-1].weight.data}")
                #if (i == 20000):
            #if (loss_values[0].item()<1e-5):
              #  ind = 0
             #   optimizer = torch.optim.LBFGS(GPT_PINN.parameters(), lr=lr_gpt)
                 #   lr_gpt=0.1*lr_gpt
                  #  optimizer = torch.optim.Adam(GPT_PINN.parameters(), lr=lr_gpt)
                    #   else:
                    #      lr_pinn=lr_gpt/0.7    
        if (i == epochs_gpt):
            print(f"layer1:{GPT_PINN.linears[0].weight.data} and {GPT_PINN.linears[0].bias.data} and layer3:{GPT_PINN.linears[-1].weight.data}")
            print("GPT-PINN Training Completed\n")         
    return loss_values[0], ep, losses, rMAE, rRMSE,loss_R,loss_IC,loss_BC,loss_R0