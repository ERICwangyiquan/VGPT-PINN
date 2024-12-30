import torch
torch.set_default_dtype(torch.float)    
from B_data import exact_u
from B_Plotting import Burgers_plot
import torch.utils.data as data
import matplotlib.pyplot as plt

def pinn1_train(PINN, nu, xt_resid, IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat,Exact_y0, epochs_pinn, lr_pinn, tol, xt_test):
    
    #ind = 1
    rMAE = [max(sum(abs(PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()]
    rRMSE = [torch.sqrt(sum((PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()]

    loss_values = PINN.loss(xt_resid,xt_test, IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat)
    losses = [loss_values[0].item()]
    ep = [0]
    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    #optimizer = optimizer.Adam(params=[{'params': PINN.parameters()},{'params':loss_weight}],lr=lr_pinn)
    #optimizer = torch.optim.LBFGS(PINN.parameters(),lr=1.0,
    #          max_iter=20,
    #          max_eval=None,
    #          tolerance_grad=1e-05,
    #          #tolerance_change=1.finfo(float).eps,
    #          tolerance_change=1e-9,
    #          history_size=100,
    #          line_search_fn="strong_wolfe",)
    batch_size = xt_resid.shape[0]
    loader = data.DataLoader(
        xt_resid,
        batch_size=batch_size,
        shuffle=True
    )

    def closure():
        optimizer.zero_grad()
        loss_values=PINN.loss(xt_batch, xt_test, IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat)[0]
        loss_values.backward()
        return loss_values


    #print(f"Epoch: 0 | Loss: {losses}")
    for i in range(1, epochs_pinn+1):
        loss_values=PINN.loss(xt_resid, xt_test, IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat) 
        
        if (loss_values[0].item() < tol):
            loss_values=PINN.loss(xt_resid, xt_test, IC_xt, IC_u, BC1, BC2,xt_RHL, xt_RHR,xt_RHt,xt_RHtL,f_hat) 
            losses.append(loss_values[0].item())
            L1_loss = max(sum(abs(PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()
            L2_loss = torch.sqrt(sum((PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()
            rMAE.append(L1_loss)
            rRMSE.append(L2_loss)
            ep.append(i)
            print(f'Epoch: {i} | Loss: {loss_values[0].item()} ,rMAE: {L1_loss}, rRMSE:{L2_loss}(Stopping Criteria Met)')
            break
        '''
        if (loss_values[0].item()<10*tol):#1e-2for1
            PINN.wi.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, PINN.parameters()), lr=lr_pinn)
            #print(PINN.wi.grad)
        loss_values=optimizer.step(closure)
        PINN.wi.data = torch.clamp(PINN.wi.data, min=0.1,max=10)
        '''  
        for step, xt_batch in enumerate(loader):
            optimizer.step(closure)
        #loss_values=optimizer.step(closure)  
        #PINN.we.data = torch.clamp(PINN.we.data, min=0)
        if (i % 100 == 0) or (i == epochs_pinn):
            #print({PINN.wi.data.item()})
            L1_loss = max(sum(abs(PINN.forward(xt_test)-Exact_y0))/sum(abs(Exact_y0))).item()
            L2_loss = torch.sqrt(sum((PINN.forward(xt_test)-Exact_y0)**2)/sum((Exact_y0)**2)).item()
            rMAE.append(L1_loss)
            rRMSE.append(L2_loss)
            losses.append(loss_values[0].item())
            ep.append(i)
            if (i % 200 == 0) or (i == epochs_pinn):
                #print(f'Epoch: {i} | loss: {loss_values[0].item()},loss_R:{round(loss_values[1].item(),3)},loss_IC:{round(loss_values[2].item(),3)},loss_BC:{round(loss_values[3].item(),3)}')
                #print(PINN.wi)
                #Burgers_plot(xt_test,abs(PINN.forward(xt_test)-Exact_y0),88,66, title=fr"PINN Solution Error $\mu={nu}$")
                print(f'Epoch: {i} | loss: {loss_values[0].item()},rMAE: {L1_loss}, rRMSE:{L2_loss},loss_RH:{loss_values[4].item():.8f},loss_con:{loss_values[5].item():.8f},loss_R:{loss_values[1].item()},loss_IC:{loss_values[2].item()},loss_BC:{loss_values[3].item()}')  
            if (i %1000 == 0):
                #if (loss_values[0].item()<1e-5):
                    #lr_pinn=0.7*lr_pinn
                    #if (L2_loss < 0.1):
                    Burgers_plot(xt_test,abs(PINN.forward(xt_test)-Exact_y0),101,201, title=fr"PINN Solution Error $\mu={nu}$")
                    Burgers_plot(xt_test,PINN.forward(xt_test),101,201, title=fr"PINN Solution $\mu={nu}$")
                    #Burgers_plot(xt_test,abs(PINN.forward(xt_test)-Exact_y0),201,501, title=fr"PINN Solution Error $\mu={nu}$")
                    #Burgers_plot(xt_test,PINN.forward(xt_test),201,501, title=fr"PINN Solution $\mu={nu}$")
                    #else:
                        #lr_pinn=lr_pinn/0.7
        if (i == 20000):
            lr_pinn = 0.1*lr_pinn
            optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)

        if (i == epochs_pinn):
            print("PINN Training Completed\n")

        #if (i == 20000) and (loss_values[0].item()>1e-3):
            #lr_pinn = 0.001
            #ind = 1
         #   tol = 1e-4
        #if (ind == 1) and (loss_values[0].item()<1e-5):
         #   optimizer = torch.optim.LBFGS(PINN.parameters(),max_iter=20, lr=0.01)
            #optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
              
    return ep,losses,  rMAE, rRMSE