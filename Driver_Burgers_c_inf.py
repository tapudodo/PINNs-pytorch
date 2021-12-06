# -*- coding: utf-8 -*-
# run: python Driver_Bugers_c_inf -- Ntrain 100
# c : continuous; inf : inference

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.PINNmodels import PhysicsInformedNN, parser,device
import torch
import torch.optim as optim     # optimizers e.g. gradient descent, ADAM, etc.


args = parser.parse_args()

#Set default dtype to float32
torch.set_default_dtype(torch.float)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
  
if __name__ == "__main__":
    noise = 0.0        
    
    # load data
    # it is a .mat file, containing 3 variables, x, t, and usol
    # x : 256*1
    # t : 100*1
    # usol : 256*100
    data = scipy.io.loadmat('data/Burgers/burgers_shock.mat') # Load data from file

    # flatten is row by row. [1,2;3,4].flatten()=[1,2,3,4]
    # [:,None] will turn it become a colomn vector

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


    '''Boundary Conditions'''
    #Initial Condition -1 =< x =<1 and t = 0  
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1 initial condition, t=0, x, 100*2
    leftedge_u = usol[:,0][:,None]# 100*1

    #Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2 boundary condition t, x=0 , 256*2
    bottomedge_u = usol[-1,:][:,None]

    #Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3 boundary condition t, x=end, 256*2
    topedge_u = usol[0,:][:,None]

    all_X_u_train = np.vstack([leftedge_x, bottomedge_x, topedge_x]) # X_u_train [456,2] (456 = 256(L1)+100(L2)+100(L3))
    all_u_train = np.vstack([leftedge_u, bottomedge_u, topedge_u])   #corresponding u [456x1]

    #choose random N_u points for training
    idx = np.random.choice(all_X_u_train.shape[0], args.Ntrain, replace=False) 

    X_u_train = all_X_u_train[idx, :] #choose indices from  set 'idx' (x,t)
    u_train = all_u_train[idx,:]      #choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    X_f_train = lb + (ub-lb)*lhs(2,args.Nftrain) 
    X_f_train = np.vstack((X_f_train, X_u_train)) # append training points to collocation points 

    'Convert to tensor and send to GPU'
    X_f_train = torch.from_numpy(X_f_train).float().to(device)
    X_u_train = torch.from_numpy(X_u_train).float().to(device)
    u_train = torch.from_numpy(u_train).float().to(device)
    X_u_test_tensor = torch.from_numpy(X_u_test).float().to(device)
    u = torch.from_numpy(u_true).float().to(device)
    f_train_zero = torch.zeros(X_f_train.shape[0],1).to(device)


    # Neural network model
    model = PhysicsInformedNN(args.nnlayers, lb, ub).to(device)

    if args.mode == 'train':

        # params = list(model.parameters())
        optimizer = optim.LBFGS(model.parameters(), lr=1e-3, 
                              max_iter = 50000, 
                              max_eval = 50000, #None, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
    
        start_time = time.time()    

        def closure():
            optimizer.zero_grad()
            loss = model.loss(X_u_train, u_train, X_f_train,f_train_zero)
            loss.backward()
            model.iter += 1
            if model.iter % 100 == 0:
                error_vec, _ = model.test(X_u_test_tensor,u)
                print(loss,error_vec)
            return loss

        optimizer.step(closure)

        'Adam Optimizer'
        # optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        # for i in range(args.epoch):
        #     loss = model.loss(X_u_train, u_train, X_f_train,f_train_zero) 
        #     optimizer.zero_grad()     # zeroes the gradient buffers of all parameters
        #     loss.backward() #backprop
        #     optimizer.step()

        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
        torch.save(model.state_dict(), 'PINNs_Burgers.pth')
    else:
        # model = PhysicsInformedNN(*args, **kwargs)
        model.load_state_dict(torch.load('PINNs_Burgers.pth') )
        model.eval()
    
    error_vec, u_pred = model.test(X_u_test_tensor,u)
    print('Test Error: %.5f'  % (error_vec))

    X_u_train = X_u_train.cpu().detach().numpy()
    u_train = u_train.cpu().detach().numpy()
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    # fig, ax = newfig(1.0, 1.1)
    fig, ax = plt.subplots()
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                  extent=[T.min(), T.max(), X.min(), X.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
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
    
    plt.savefig('figures/Burgers',dpi = 800)  