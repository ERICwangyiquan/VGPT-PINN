import torch
#torch.set_default_dtype(torch.float)    
#from torch.optim import LBFGS
from E_Plotting import E_plot
import matplotlib.pyplot as plt
import torch.utils.data as data



def gpt_train(VGPT_PINN, nu, xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test, xt_en, xt_RH, xt_RHL,weno_u,epochs_tgpt_list, lr_tgpt, tol_tgpt):

    batch_size = 1000
    loader = data.DataLoader(
        xt_resid,
        batch_size=batch_size,
        shuffle=True
    )
    def closure():
        optimizer.zero_grad()
        WE_loss_values=VGPT_PINN.loss(xt_batch, IC_xt, IC_u,BC_xt, BC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
        WE_loss_values.backward()
        return WE_loss_values
        
    change_tol = 100*tol_tgpt
    optim1_epoch_tgpt = epochs_tgpt_list[0]
    epochs_tgpt =epochs_tgpt_list[1]
    #optim1_epoch_tgpt = int(epochs_tgpt/2)
    WE_loss_values = VGPT_PINN.loss_align(xt_resid, IC_xt,IC_u,BC_xt, BC_u, f_hat,xt_test,xt_en, xt_RH, xt_RHL)
    WE_losses=[WE_loss_values.item()]
    ep = [0]
    #E_plot(xt_test, VGPT_PINN.forward(xt_test)[0], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $\rho$ Solution at 0 step")
    #E_plot(xt_test, VGPT_PINN.forward(xt_test)[1], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $P$ Solution at 0 step")
    #E_plot(xt_test, VGPT_PINN.forward(xt_test)[2], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $V$ Solution at 0 step")
    print(f'{nu}: {VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
    #print(f'{VGPT_PINN.linears[0].weight.data}, {VGPT_PINN.linears[0].bias.data} and {VGPT_PINN.linears[-3].weight.data.item()},{VGPT_PINN.linears[-2].weight.data.item()},{VGPT_PINN.linears[-1].weight.data.item()}')
    #print(f' Loss: {WE_loss_values[0].item():.6f},LossR: {WE_loss_values[1].item():.6f}, lossIC:{WE_loss_values[2].item()} (Stopping Criteria Met)')
    print(f'{nu}: Loss: {WE_loss_values.item():.6f}')
    optimizer = torch.optim.Adam(VGPT_PINN.parameters(), lr=lr_tgpt)

    print(f"{nu}: Epoch: 0 | Loss: {WE_losses[0]}")
    for i in range(1, min(epochs_tgpt,optim1_epoch_tgpt)+1):
        if (WE_loss_values.item() < tol_tgpt):
            WE_losses.append(WE_loss_values.item())
            ep.append(i)
            #print(f'{VGPT_PINN.linears[0].weight.grad}, {VGPT_PINN.linears[0].bias.data} and {VGPT_PINN.linears[-3].weight.data.item()},{VGPT_PINN.linears[-2].weight.data.item()},{VGPT_PINN.linears[-1].weight.data.item()}')
            print(f'{nu}:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
            print(f'Epoch: {i} | Loss: {WE_loss_values.item():.6f} (VGPT_PINN1 Tol Criteria Met)')
            break
            
        optimizer.zero_grad()
        WE_loss_values=VGPT_PINN.loss_align(xt_resid, IC_xt, IC_u,BC_xt, BC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
        WE_loss_values.backward()
        optimizer.step()
        #print(f'{nu}:Epoch: {i} | Loss: {WE_loss_values.item():.6f}')
        if (i % 100 == 0) or (i == epochs_tgpt):
            WE_losses.append(WE_loss_values.item())
            ep.append(i)
            if (i % 500 == 0) or (i == epochs_tgpt):
                #E_plot(xt_test, VGPT_PINN.forward(xt_test)[0], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $\rho$ Solution at {i} step")
                #E_plot(xt_test, VGPT_PINN.forward(xt_test)[1], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $P$ Solution at {i} step")
                #E_plot(xt_test, VGPT_PINN.forward(xt_test)[2], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $V$ Solution at {i} step")
                #print(f'{nu}:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
                #print(f'{nu[1]}: {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
                #print(f'{VGPT_PINN.linears[0].weight.data}, {VGPT_PINN.linears[0].bias.data} and {VGPT_PINN.linears[-3].weight.data.item()},{VGPT_PINN.linears[-2].weight.data.item()},{VGPT_PINN.linears[-1].weight.data.item()}')
                print(f'{nu}:Epoch: {i} | Loss: {WE_loss_values.item():.6f}')
                '''
                plt.figure(dpi=150, figsize=(5,4))
                Nx=100
                ind = 100
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[0][ind-1::Nx].detach().cpu(),label=fr"$Density$")
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[1][ind-1::Nx].detach().cpu(),label=fr"$Pressure$")
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[2][ind-1::Nx].detach().cpu(),label=fr"$Velocity$")
                plt.legend(fontsize = 12)
                plt.show()
                '''
                #if (i % 2000 == 0):
                 #   lr_tgpt=0.7*lr_tgpt
                if (i == min(epochs_tgpt,optim1_epoch_tgpt)):
                    print(f'{nu}:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
                    print(f'Loss: {WE_loss_values.item():.6f} (VGPT_PINN1 Step Criteria Met)\n')

    plt.figure(dpi=150, figsize=(5,4))
    Nx=100
    ind = 100
    plt.plot(weno_u[3],weno_u[0],'k')#,label="$Density$"
    plt.plot(weno_u[3],weno_u[1],'k')#,label="$Pressure$"
    plt.plot(weno_u[3],weno_u[2],'k')#,label="$velocity$"

    plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[0][ind-1::Nx].detach().cpu(),'--',label=fr"$Density$")
    plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[1][ind-1::Nx].detach().cpu(),'--',label=fr"$Pressure$")
    plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[2][ind-1::Nx].detach().cpu(),'--',label=fr"$Velocity$")
    plt.legend(fontsize = 12)
    plt.show()
    
    print(f"Step 2 Training Begin...")

    VGPT_PINN.linears[-3].weight.requires_grad = True
    VGPT_PINN.linears[-2].weight.requires_grad = True
    VGPT_PINN.linears[-1].weight.requires_grad = True
    params = [
    {'params': VGPT_PINN.linears[-3].parameters(), 'lr': lr_tgpt*100},
    {'params': VGPT_PINN.linears[-2].parameters(), 'lr': lr_tgpt*100},
    {'params': VGPT_PINN.linears[-1].parameters(), 'lr': lr_tgpt*100}
]
    #for name, param in VGPT_PINN.named_parameters():
    #    if name not in ['linears.5.weight', 'linears.6.weight','linears.7.weight']:
    #        print(name)
    #other_params = [param for name, param in VGPT_PINN.named_parameters() if name not in ['linears.5.weight', 'linears.6.weight','linears.7.weight']]
    #other_params = [param for name, param in VGPT_PINN.named_parameters() if int(name.split('.')[-2]) < -3]
    other_params = [param for name, param in VGPT_PINN.named_parameters() if int(name.split('.')[-2]) < int((len(list(VGPT_PINN.named_parameters()))-3)/2)]
    params.append({'params': other_params, 'lr': lr_tgpt})
    optimizer = torch.optim.Adam(params)
    '''
    for name, param in VGPT_PINN.named_parameters():
        if param.requires_grad:
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is param:
                        print(f"Parameter: {name}, Learning Rate: {group['lr']}")
    '''
    #lr_pinn=0.0001
    #optimizer = torch.optim.LBFGS(VGPT_PINN.parameters(),lr=0.01,max_iter=20 )
    for i in range(optim1_epoch_tgpt+1, epochs_tgpt+1):
        #WE_loss_values = PINN.loss(xt_resid, IC_xt,IC_u, f_hat)
        #PINN.train()
        if (WE_loss_values.item() < tol_tgpt) or (i == epochs_tgpt):
            WE_losses.append(WE_loss_values.item())
            ep.append(i)
            print(f'{nu}:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
            #print(f'{VGPT_PINN.linears[0].weight.grad}, {VGPT_PINN.linears[0].bias.data} and {VGPT_PINN.linears[-3].weight.data.item()},{VGPT_PINN.linears[-2].weight.data.item()},{VGPT_PINN.linears[-1].weight.data.item()}')
            print(f'{nu}:Epoch: {i} | Loss: {WE_loss_values.item():.6f} (VGPT_PINN2 Step Criteria Met)\n')
            break
        #WE_loss_values = optimizer.step(closure)
        
        for step, xt_batch in enumerate(loader):
            WE_loss_values = optimizer.step(closure)
            #print(f'{step}:Loss:{WE_loss_values}')
        '''
        optimizer.zero_grad()
        WE_loss_values=VGPT_PINN.loss(xt_batch, IC_xt, IC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
        #print(f'{nu}:Epoch: {step} | Loss: {WE_loss_values.item():.6f}')
        WE_loss_values.backward()
        optimizer.step()  
        optimizer.zero_grad()
        WE_loss_values=VGPT_PINN.loss(xt_resid, IC_xt, IC_u,f_hat,xt_test,xt_en, xt_RH, xt_RHL)
        WE_loss_values.backward()
        optimizer.step()
        '''
        if (i % 10 == 0) or (i == epochs_tgpt):
            WE_losses.append(WE_loss_values.item())
            ep.append(i)
            if (i % 5000 == 0) or (i == epochs_tgpt):
                #E_plot(xt_test, VGPT_PINN.forward(xt_test)[0], dpi=80, figsize=(5,4),title=fr"VGPT-PINN $\rho$ Solution at {i} step")
                print(f'{nu}:{VGPT_PINN.linears[0].weight.grad} and {VGPT_PINN.linears[0].weight.data}')
                print(f'{nu}:{VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
                #print(f'{VGPT_PINN.linears[0].weight.grad}, {VGPT_PINN.linears[0].bias.data} and {VGPT_PINN.linears[-3].weight.data.item()},{VGPT_PINN.linears[-2].weight.data.item()},{VGPT_PINN.linears[-1].weight.data.item()}')
                print(f'{nu}:Epoch: {i} | Loss: {WE_loss_values.item():.6f}')
                plt.figure(dpi=150, figsize=(5,4))
                Nx=100
                ind = 100
                plt.plot(weno_u[3],weno_u[0],'k')#,label="$Density$"
                plt.plot(weno_u[3],weno_u[1],'k')#,label="$Pressure$"
                plt.plot(weno_u[3],weno_u[2],'k')#,label="$velocity$"
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[0][ind-1::Nx].detach().cpu(),'--',label=fr"$Density$")
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[1][ind-1::Nx].detach().cpu(),'--',label=fr"$Pressure$")
                plt.plot(xt_test[0::Nx,1].detach().cpu(),VGPT_PINN(xt_test)[2][ind-1::Nx].detach().cpu(),'--',label=fr"$Velocity$")
                plt.legend(fontsize = 12)
                plt.show()
                '''
                if (WE_loss_values.item() < change_tol):
                    lr_tgpt=0.1*lr_tgpt
                    params = [
                    {'params': VGPT_PINN.linears[-3].parameters(), 'lr': lr_tgpt*100},
                    {'params': VGPT_PINN.linears[-2].parameters(), 'lr': lr_tgpt*100},
                    {'params': VGPT_PINN.linears[-1].parameters(), 'lr': lr_tgpt*100}
                ]
                    change_tol = 0.1*change_tol
                    print(f'lr:{lr_tgpt*100} and change_tol:{change_tol}')
                '''
                if (i == epochs_tgpt):
                    print(f'{nu}:{VGPT_PINN.linears[0].weight.data} and {VGPT_PINN.linears[-3].weight.data},{VGPT_PINN.linears[-2].weight.data},{VGPT_PINN.linears[-1].weight.data}')
                    print(f'Loss: {WE_loss_values.item():.6f} (VGPT_PINN2 Step Criteria Met)\n')
                    
    return WE_loss_values, ep, WE_losses
