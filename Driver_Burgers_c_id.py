# -*- coding: utf-8 -*-
# run: python Driver_Bugers_c_id -- Ntrain 2000
# c : continuous; id : identifier

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.PINNmodels import PhysicsInformedNN, parser,device
import torch
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from src.plotting import newfig, savefig

# from scipy.interpolate import griddata
# from mpl_toolkits.mplot3d import Axes3D

args = parser.parse_args()

#Set default dtype to float32
torch.set_default_dtype(torch.float)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

class BurgersPara():
    def __init__(self,lambda1,lambda2):
        # settings
        self.lambda1 = torch.tensor(lambda1, requires_grad=True).to(device)
        self.lambda2 = torch.tensor(lambda2, requires_grad=True).to(device)
        
        self.lambda1 = torch.nn.Parameter(self.lambda1)
        self.lambda2 = torch.nn.Parameter(self.lambda2)
    
if __name__ == "__main__": 
    # Setting
    nu = 0.01/np.pi
    
    # Data import
    data = scipy.io.loadmat('data/Burgers/burgers_shock.mat')
    x = data['x']                                   # 256 points between -1 and 1 [256x1]
    t = data['t']                                   # 100 time points between 0 and 1 [100x1] 
    usol = data['usol']                             # solution of 256x100 grid points
    
    X, T = np.meshgrid(x,t) # X : 256*100

    X_u_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    '''
        Fortran Style ('F') flatten,stacked column wise!
        u = [c1 
            c2
            .
            .
            cn]
        u =  [25600x1] 
    '''
    u_true = usol.flatten('F')[:,None]

    # Domain bounds
    lb = X_u_test[0]  # [-1. 0.]
    ub = X_u_test[-1] # [1.  0.99]

    ######################################################################
    ######################## Noiseless Data ##############################
    ######################################################################
    noise = 0.0            
             
    idx = np.random.choice(X_u_test.shape[0], args.Ntrain, replace=False)
    X_u_train = X_u_test[idx,:]
    u_train = u_true[idx,:]


    'Convert to tensor and send to GPU'
    X_u_train = torch.from_numpy(X_u_train).float().to(device)
    u_train = torch.from_numpy(u_train).float().to(device)
    X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)
    u = torch.from_numpy(u_true).float().to(device)
    u_train_zero = torch.zeros(u_train.shape[0],1).to(device)
    
    # Neural network model
    model = PhysicsInformedNN(args.nnlayers, lb, ub).to(device)

    BurgersParamodel = BurgersPara([0.0],[-6.0])

    model.register_parameter('lambda1', BurgersParamodel.lambda1)
    model.register_parameter('lambda2', BurgersParamodel.lambda2)
    
    if args.mode == 'train':

        # params = list(model.parameters())
        optimizer = optim.LBFGS(model.parameters(), lr=1, 
                              max_iter = 50000,
                              line_search_fn = 'strong_wolfe')
        'Adam Optimizer'
        optimizerAdam = torch.optim.Adam(model.parameters())
        start_time = time.time()  

        # model.train()
        for epoch in range(0):
            optimizerAdam.zero_grad()
            loss = model.loss_id(X_u_train, u_train,u_train_zero,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
            # Backward and optimize
            loss.backward()
            optimizerAdam.step()
            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' % 
                    (
                        epoch, 
                        loss.item(), 
                        BurgersParamodel.lambda1.item(), 
                        torch.exp(BurgersParamodel.lambda2.detach()).item()
                    )
                )
        # Backward and optimize
        def closure():
            optimizer.zero_grad()
            loss_func = model.loss_id(X_u_train, u_train, u_train_zero,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
            loss_func.backward()
            model.iter += 1
            if model.iter % 100 == 0:
                # error_vec, _ = model.test(X_u_test_tensor,u)
                print(
                'Loss: %e, l1: %.5f, l2: %.5f' % 
                    (
                    loss_func.item(), 
                    BurgersParamodel.lambda1.item(), 
                    torch.exp(BurgersParamodel.lambda2.detach()).item()
                    )
                )
            return loss_func

        optimizer.step(closure)
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))


        torch.save(model.state_dict(), 'PINNs_Burgers_iden_noiseless.pth')
        
        
        np.savetxt('BurgersPara_iden_noiseless.out', ( BurgersParamodel.lambda1.cpu().detach().numpy() , BurgersParamodel.lambda2.cpu().detach().numpy() ) )
    else:
        # model = PhysicsInformedNN(*args, **kwargs)
        model.load_state_dict(torch.load('PINNs_Burgers_iden_noiseless.pth') )
        lambda1, lambda2 = np.loadtxt('BurgersPara_iden_noiseless.out', unpack=True)
        BurgersParamodel = BurgersPara([lambda1], [lambda2])
        model.eval()


    f_pred = model.forward_f(X_u_test_tensor,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
    error_vec, u_pred = model.test(X_u_test_tensor,u)
    
    lambda1_value = BurgersParamodel.lambda1.detach().cpu().numpy()
    lambda2_value = BurgersParamodel.lambda2.detach().cpu().numpy()
    lambda2_value = np.exp(lambda2_value)
    
    error_lambda1 = np.abs(lambda1_value - 1.0)*100
    error_lambda2 = np.abs(lambda2_value - nu)/nu * 100
    
    print('Error u: %e' % (error_vec))    
    print('Error l1: %.5f%%' % (error_lambda1))                             
    print('Error l2: %.5f%%' % (error_lambda2))  
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01        
    u_train = u_train.cpu().detach().numpy()
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    u_train = torch.from_numpy(u_train).float().to(device)


    # Neural network model
    model = PhysicsInformedNN(args.nnlayers, lb, ub).to(device)

    BurgersParamodel = BurgersPara([0.0],[-6.0])

    model.register_parameter('lambda1', BurgersParamodel.lambda1)
    model.register_parameter('lambda2', BurgersParamodel.lambda2)
    
    if args.mode == 'train':

        # params = list(model.parameters())
        optimizer = optim.LBFGS(model.parameters(), lr=1, 
                              max_iter = 50000,
                              line_search_fn = 'strong_wolfe')
        'Adam Optimizer'
        optimizerAdam = torch.optim.Adam(model.parameters())
        start_time = time.time()  

        model.train()
        for epoch in range(args.epoch):
            optimizerAdam.zero_grad()
            loss = model.loss_id(X_u_train, u_train,u_train_zero,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
            # Backward and optimize
            loss.backward()
            optimizerAdam.step()
            if epoch % 100 == 0:
                print(
                    'It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' % 
                    (
                        epoch, 
                        loss.item(), 
                        BurgersParamodel.lambda1.item(), 
                        torch.exp(BurgersParamodel.lambda2.detach()).item()
                    )
                )
        # Backward and optimize
        def closure():
            optimizer.zero_grad()
            loss_func = model.loss_id(X_u_train, u_train, u_train_zero,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
            loss_func.backward()
            model.iter += 1
            if model.iter % 100 == 0:
                # error_vec, _ = model.test(X_u_test_tensor,u)
                print(
                'Loss: %e, l1: %.5f, l2: %.5f' % 
                    (
                    loss_func.item(), 
                    BurgersParamodel.lambda1.item(), 
                    torch.exp(BurgersParamodel.lambda2.detach()).item()
                    )
                )
            return loss_func

        optimizer.step(closure)
        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))


        torch.save(model.state_dict(), 'PINNs_Burgers_iden_noisy.pth')
        np.savetxt('BurgersPara_iden_noisy.out', ( BurgersParamodel.lambda1.cpu().detach().numpy() , BurgersParamodel.lambda2.cpu().detach().numpy() ) )
    else:
        # model = PhysicsInformedNN(*args, **kwargs)
        model.load_state_dict(torch.load('PINNs_Burgers_iden_noisy.pth') )
        lambda1, lambda2 = np.loadtxt('BurgersPara_iden_noisy.out', unpack=True)
        BurgersParamodel = BurgersPara([lambda1], [lambda2])
        model.eval()

    f_pred = model.forward_f(X_u_test_tensor,BurgersParamodel.lambda1,BurgersParamodel.lambda2)
    error_vec, u_pred = model.test(X_u_test_tensor,u)
    
    lambda1_value_noisy = BurgersParamodel.lambda1.detach().cpu().numpy()
    lambda2_value_noisy = BurgersParamodel.lambda2.detach().cpu().numpy()
    lambda2_value_noisy = np.exp(lambda2_value_noisy)
    
    error_lambda1_noisy = np.abs(lambda1_value_noisy - 1.0)*100
    error_lambda2_noisy = np.abs(lambda2_value_noisy - nu)/nu * 100
    
    print('Error u: %e' % (error_vec))    
    print('Error lambda_1: %.5f%%' % (error_lambda1_noisy))                             
    print('Error lambda_2: %.5f%%' % (error_lambda2_noisy))                          

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    # fig, ax = newfig(1.0, 1.4)
    fig, ax = plt.subplots()
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)
    
    ####### Row 3: Identified PDE ##################    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
    
    ax = plt.subplot(gs2[:, :])
    ax.axis('off')

    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda1_value, lambda2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda1_value_noisy, lambda2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(0.1,0.1,s)
        
    savefig('figures/Burgers_identification')  
    



