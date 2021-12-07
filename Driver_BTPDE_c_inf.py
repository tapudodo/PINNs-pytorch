# -*- coding: utf-8 -*-
# run: python Driver_Bugers_c_inf -- Ntrain 100
# c : continuous; inf : inference

import numpy as np
import scipy.io
from pyDOE import lhs
import time
import torch
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from src.PINNBTPDEmodels import PINNBTPDENN, parser,device
from src.CoordTransform import Square2Circle, Circle2Square,dxi_eta
from src.samplegrid import samplegrid, samplegrid_v2

args = parser.parse_args()

#Set default dtype to float32
torch.set_default_dtype(torch.float)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
  
if __name__ == "__main__":
    radius = args.radius
    demiheight = args.demiheight
    Coord_xi = np.linspace(-radius,radius,args.dxi*radius+1)
    Coord_eta = np.linspace(-radius,radius,args.deta*radius+1)
    Coord_z = np.linspace(-demiheight,demiheight,args.dz*demiheight+1)
    # t = np.hstack((np.linspace(0,args.delta,args.delta/args.dt,endpoint=False),
    #                 np.linspace(args.delta,args.Delta,args.diff_delta/args.dt,endpoint=False),
    #                 np.linspace(args.Delta,args.Delta+args.delta,args.delta/args.dt+1) ))
    t = np.linspace(0,args.delta+args.Delta,(args.delta+args.Delta)//args.dt+1)

    #Initial Condition t = 0, Input should be N_sample_ic*4, and the 4 lines should be 0
    Input_IC_train = samplegrid((Coord_xi,Coord_eta,Coord_z,t),args.N_IC_train,3,0)

    Input_IC_train = torch.from_numpy(Input_IC_train).float().to(device)
    Ones_IC_train = torch.ones(Input_IC_train.shape[0],1).to(device)

    '''Boundary Conditions'''
    Input_BC_train = np.vstack ( ( samplegrid_v2((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,0,-radius), 
                                    samplegrid_v2((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,0,radius), 
                                    samplegrid_v2((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,1,-radius), 
                                    samplegrid_v2((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,1,radius) ) )

    Input_BC_train_circle = np.copy(Input_BC_train)
    Input_BC_train_circle[:,[0]],Input_BC_train_circle[:,[1]] = Square2Circle(Input_BC_train_circle[:,[0]],Input_BC_train_circle[:,[1]],radius)
    dxieta_xy_BC = dxi_eta(Input_BC_train_circle[:,[0]],Input_BC_train_circle[:,[1]],radius)

    Input_BC_train = torch.from_numpy(Input_BC_train).float().to(device)
    Zero_BC_train = torch.zeros(Input_BC_train.shape[0],1).to(device)
    Input_BC_train_circle = torch.from_numpy(Input_BC_train_circle/radius).float().to(device)
    dxieta_xy_BC = (torch.from_numpy(dxieta_xy_BC[0]).float().to(device) ,
                    torch.from_numpy(dxieta_xy_BC[1]).float().to(device),
                    torch.from_numpy(dxieta_xy_BC[2]).float().to(device),
                    torch.from_numpy(dxieta_xy_BC[3]).float().to(device))

    Input_BC_top_bottom_train = np.vstack ( ( samplegrid((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,2,-demiheight), 
                                    samplegrid((Coord_xi,Coord_eta,Coord_z,t),args.N_BC_train,2,demiheight) ) )

    Input_BC_top_bottom_train = torch.from_numpy(Input_BC_top_bottom_train).float().to(device)
    Zero_BC_top_bottom_train = torch.zeros(Input_BC_top_bottom_train.shape[0],1).to(device)

    '''Collocation Points'''
    # Latin Hypercube sampling for collocation points 
    # N_f sets of tuples(x,t)
    # Domain bounds
    lb = np.array([-radius,-radius,-demiheight,0])
    ub = np.array([radius,radius,demiheight,args.delta+args.Delta])
    Input_f_train = lb + (ub-lb)*lhs(4,args.N_f_train)
    Input_f_train_circle = np.copy(Input_f_train)
    Input_f_train_circle[:,[0]],Input_f_train_circle[:,[1]] = Square2Circle(Input_f_train_circle[:,[0]],Input_f_train_circle[:,[1]],radius)
    dxieta_xy_f = dxi_eta(Input_f_train_circle[:,[0]],Input_f_train_circle[:,[1]],radius)

    'Convert to tensor and send to GPU'
    Input_f_train = torch.from_numpy(Input_f_train).float().to(device)
    Zero_f_train = torch.zeros(Input_f_train.shape[0],1).to(device)
    Input_f_train_circle = torch.from_numpy(Input_f_train_circle).float().to(device)
    dxieta_xy_f = (torch.from_numpy(dxieta_xy_f[0]).float().to(device) ,
                    torch.from_numpy(dxieta_xy_f[1]).float().to(device),
                    torch.from_numpy(dxieta_xy_f[2]).float().to(device),
                    torch.from_numpy(dxieta_xy_f[3]).float().to(device))

    ##################### create test set#########################
    # load data
    # massmatrix = scipy.io.loadmat('data/BTPDE/massmatrix.mat') # Load data from file
    # femesh = scipy.io.loadmat('data/BTPDE/femesh.mat') # Load data from file
    # # femesh struct: 
    # # ncompartment = 1
    # # nboundary = 2
    # # points = 3*9233
    # # facets = 2 cells, 3*n, 3*n
    # # elements = 4*31490
    # # point_map useless

    # X_u_test_tensor = torch.from_numpy(X_test).float().to(device)
    # u = torch.from_numpy(u_true).float().to(device)


    # Neural network model
    model = PINNBTPDENN(args.nnlayers,args.diffusivity,args.delta,args.Delta,args.bvalue,[[0],[0],[1]]).to(device)

    if args.mode == 'train':

        # params = list(model.parameters())
        optimizer = optim.LBFGS(model.parameters(), lr=1e-3, 
                              max_iter = 5001, 
                              max_eval = 5001, #None, 
                              tolerance_grad = 1e-05, 
                              tolerance_change = 1e-09, 
                              history_size = 100, 
                              line_search_fn = 'strong_wolfe')
        'Adam Optimizer'
        optimizerAdam = torch.optim.Adam(model.parameters())
        start_time = time.time()    

        model.train()
        for epoch in range(args.epoch):
            optimizerAdam.zero_grad()
            loss = model.loss(Input_IC_train,Ones_IC_train,
                                Input_BC_train,Zero_BC_train,dxieta_xy_BC,Input_BC_train_circle,
                                Input_BC_top_bottom_train,Zero_BC_top_bottom_train,
                                Input_f_train,Zero_f_train,dxieta_xy_f,Input_f_train_circle)
            # Backward and optimize
            loss.backward()
            optimizerAdam.step()
            if epoch % 10 == 0:
                print(
                    'It: %d, Loss: %.3e' % 
                    (
                        epoch, 
                        loss.item()
                    )
                )


        def closure():
            optimizer.zero_grad()
            loss = model.loss(Input_IC_train,Ones_IC_train,
                                Input_BC_train,Zero_BC_train,dxieta_xy_BC,Input_BC_train_circle,
                                Input_BC_top_bottom_train,Zero_BC_top_bottom_train,
                                Input_f_train,Zero_f_train,dxieta_xy_f,Input_f_train_circle)
            loss.backward()
            model.iter += 1
            if model.iter % 10 == 0:
                # error_vec, _ = model.test(X_test,u)
                # print(loss,error_vec)
                print(loss)
            return loss

        optimizer.step(closure)


        elapsed = time.time() - start_time                
        print('Training time: %.4f' % (elapsed))
        torch.save(model.state_dict(), 'PINNs_BTPDE.pth')
    else:
        # model = PINNBTPDENN(*args, **kwargs)
        model.load_state_dict(torch.load('PINNs_BTPDE.pth') )
        model.eval()
    
    # error_vec, u_pred = model.test(X_test,u)
    # print('Test Error: %.5f'  % (error_vec))