# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn                     # neural networks
import torch.autograd as autograd         # computation graph
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--lr', default=1e-3, type=float) # learning rate
parser.add_argument('--load', default=True, type=bool) # load model before training
parser.add_argument('--random_seed', default=1234, type=int)
parser.add_argument('--epoch', default=2001, type=int)
parser.add_argument('--Ntrain', default=400,type=int) # train set. for inden, 2000. for inference 100
parser.add_argument('--Nftrain', default=5000,type=int) # train set. 
parser.add_argument("--nnlayers", default=[2, 100, 100, 100, 100, 100, 100, 100, 100, 1])

# parser.add_argument('--input_dim', default=4, type=int) # input dimension = spatial + temporal
# parser.add_argument('--output_dim', default=1, type=int) # output dimension = 1

# parser.add_argument('--lr_pretrain', default=3e-4, type=float) # not sure whether we need to pretrain
# parser.add_argument('--epoch_pretrain', default=30001, type=int)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PhysicsInformedNN(nn.Module):
    def __init__(self,layerslist,ub,lb):
        super(PhysicsInformedNN, self).__init__()
        #input dim: 2 variables, spatial x and temporal t
        #output dim: 1. u    
        # [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]  
        self.lb = torch.from_numpy(lb).float().to(device)
        self.ub = torch.from_numpy(ub).float().to(device)

        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')

        self.nu = 0.01/np.pi

        # self.flatten = nn.Flatten()
        modules = []
        for i in range(0,len(layerslist)-1):
            nnlayer = nn.Linear(layerslist[i], layerslist[i+1])
            nn.init.xavier_normal_(nnlayer.weight.data)
            nn.init.zeros_(nnlayer.bias.data)
            modules.append(nnlayer)
            if i != len(layerslist)-2:
                modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)
        # alternative choice is nn.init.normal_ or nn.init.kaiming_uniform_ kaiming_normal_

        # hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        # last_layer = nn.Linear(hidden_layer_sizes[-1], 1)
        # self.layers =\
        #     [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
        #      for input_, output_ in 
        #      zip(hidden_layer_sizes, hidden_layer_sizes[1:])] +\
        #     [last_layer]
        
        # # The output activation depends on the problem
        # if sigmoid:
        #     self.layers = self.layers + [nn.Sigmoid()]
            
        # self.layers = nn.Sequential(*self.layers)


        self.iter = 0


    def forward(self, X):
        # X is stack of x and t
        if torch.is_tensor(X) != True:         
            x = torch.from_numpy(X)

        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        H = H.float()
        # X = self.flatten(H)
        H = self.layers(H)
        return H

    def loss_BC(self,x,y):
                
        loss_u = self.loss_function(self.forward(x), y)
                
        return loss_u
    
    def loss_PDE(self, x_to_train_f,f_train_zero):
                
        # x_1_f = x_to_train_f[:,[0]]
        # x_2_f = x_to_train_f[:,[1]]
                        
        g = x_to_train_f.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)
                
        u_x_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
                                                            
        u_x = u_x_t[:,[0]]
        
        u_t = u_x_t[:,[1]]
        
        u_xx = u_xx_tt[:,[0]]
                                        
        f = u_t + (self.forward(g))*(u_x) - (self.nu)*u_xx 
        
        loss_f = self.loss_function(f,f_train_zero)
                
        return loss_f

    def forward_f(self, X, lambda1, lambda2):
        """ The pytorch autograd version of calculating residual """
        lambda_1 = lambda1        
        lambda_2 = torch.exp(lambda2)
        u = self.forward(X)
        
        # x_1_f = X[:,[0]]
        # x_2_f = X[:,[1]]
                        
        g = X.clone()
                        
        g.requires_grad = True
        
        u = self.forward(g)
                
        u_x_t = autograd.grad(u,g,torch.ones([X.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
                                
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(X.shape).to(device), create_graph=True)[0]
                                                            
        u_x = u_x_t[:,[0]]
        
        u_t = u_x_t[:,[1]]
        
        u_xx = u_xx_tt[:,[0]]
        
        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx

        return f

    def loss_PDE_id(self, X, u_train_zero, lambda1, lambda2):
        f = self.forward_f(X, lambda1, lambda2)
        loss_f = self.loss_function(f,u_train_zero)
        return loss_f
    
    def loss(self,x,y,x_to_train_f,f_train_zero):

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(x_to_train_f,f_train_zero)
        
        loss_val = loss_u + loss_f
        
        return loss_val

    def loss_id(self,x,y,u_train_zero,lambda1,lambda2):

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE_id(x,u_train_zero,lambda1,lambda2)
        
        loss_val = loss_u + loss_f
        
        return loss_val
    
    def predict(self, X):
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)
        u_pred = self(X).cpu().detach().numpy().squeeze()
        # u_pred = np.reshape(u_pred,(256,100),order='F')
        return u_pred

    def test(self, X_u_test_tensor, u):
                
        u_pred = self.forward(X_u_test_tensor)
        
        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
        
        u_pred = u_pred.cpu().detach().numpy()
        
        u_pred = np.reshape(u_pred,(256,100),order='F')
                
        return error_vec, u_pred