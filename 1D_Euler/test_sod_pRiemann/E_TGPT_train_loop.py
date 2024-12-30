import torch
#torch.set_default_dtype(torch.float)    
#from torch.optim import LBFGS
from E_Plotting import E_plot,E_plot_ut
import matplotlib.pyplot as plt
import torch.utils.data as data



def gpt_train(TGPT_PINN, nu, xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test, xt_en, xt_RH, xt_RHL,weno_u,epochs_tgpt_list, lr_tgpt, tol_tgpt):

    def closure():
        optimizer.zero_grad()
        WE_loss_values=TGPT_PINN.loss(xt_resid, IC_xt, IC_u,BC_xt, BC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
        WE_loss_values.backward()
        return WE_loss_values
        
    single_loop = epochs_tgpt_list[0]
    epochs_tgpt =epochs_tgpt_list[1]
    WE_loss_values = TGPT_PINN.loss(xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test,xt_en, xt_RH, xt_RHL)
    PDE_losses = [WE_loss_values[0].item()]
    WE_losses=[WE_loss_values[1].item()]
    ep = [0]
    #print(f'{nu}: {TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[-3].weight.data},{TGPT_PINN.linears[-2].weight.data},{TGPT_PINN.linears[-1].weight.data}')
    print(f'{nu}: WE-Loss: {WE_loss_values[1].item():.6f},PDE-Loss: {WE_loss_values[0].item():.6f}')


    for loop in range(0,int(epochs_tgpt/single_loop)):
        print(f"Stage {loop} Training Begin...")
        if (loop % 2 == 0):
            TGPT_PINN.linears[-3].weight.requires_grad = False
            TGPT_PINN.linears[-2].weight.requires_grad = False
            TGPT_PINN.linears[-1].weight.requires_grad = False
            optimizer = torch.optim.Adam(TGPT_PINN.parameters(), lr=lr_tgpt)

        else:
            TGPT_PINN.linears[-3].weight.requires_grad = True
            TGPT_PINN.linears[-2].weight.requires_grad = True
            TGPT_PINN.linears[-1].weight.requires_grad = True
            params = [
            {'params': TGPT_PINN.linears[-3].parameters(), 'lr': lr_tgpt*100},
            {'params': TGPT_PINN.linears[-2].parameters(), 'lr': lr_tgpt*100},
            {'params': TGPT_PINN.linears[-1].parameters(), 'lr': lr_tgpt*100}
        ]
            other_params = [param for name, param in TGPT_PINN.named_parameters() if int(name.split('.')[-2]) < int((len(list(TGPT_PINN.named_parameters()))-3)/2)]
            params.append({'params': other_params, 'lr': lr_tgpt})
            optimizer = torch.optim.Adam(params)

        print(f'{nu}:{TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[-3].weight.data},{TGPT_PINN.linears[-2].weight.data},{TGPT_PINN.linears[-1].weight.data}')
        for i in range(loop*single_loop+1, (loop+1)*single_loop+1):
            WE_loss_values=TGPT_PINN.loss(xt_resid, IC_xt, IC_u,BC_xt, BC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
            if (WE_loss_values[1].item() < tol_tgpt):
                PDE_losses.append(WE_loss_values[0].item())
                WE_losses.append(WE_loss_values[1].item())
                ep.append(i)
                print(f'{nu}:{TGPT_PINN.linears[0].weight.data} and {TGPT_PINN.linears[-3].weight.data},{TGPT_PINN.linears[-2].weight.data},{TGPT_PINN.linears[-1].weight.data}')
                print(f'Epoch: {i} | WE-Loss: {WE_loss_values[1].item():.6f},PDE-Loss: {WE_loss_values[0].item():.6f} (TGPT_PINN Tol Criteria Met)')
                break
            else:
                optimizer.zero_grad()
                WE_loss_values[1].backward()
                optimizer.step()

                if (i % 10 == 0) or (i == epochs_tgpt):
                    PDE_losses.append(WE_loss_values[0].item())
                    WE_losses.append(WE_loss_values[1].item())
                    ep.append(i)
                    if (i % 200 == 0) or (i == epochs_tgpt):
                        print(f'{nu}:Epoch: {i} | WE-Loss: {WE_loss_values[1].item():.6f},PDE-Loss: {WE_loss_values[0].item():.6f}')
                        #E_plot_ut(xt_test,TGPT_PINN(xt_test),weno_u)

        E_plot_ut(xt_test,TGPT_PINN(xt_test),weno_u)
                    
    return WE_loss_values, ep, WE_losses,PDE_losses
