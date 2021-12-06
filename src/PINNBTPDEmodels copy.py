# -*- coding: utf-8 -*-

import argparse
import torch
from torch._C import unify_type_list
import torch.nn as nn                     # neural networks
import torch.autograd as autograd         # computation graph
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
parser.add_argument('--lr', default=1e-3, type=float) # learning rate
parser.add_argument('--load', default=True, type=bool) # load model before training
parser.add_argument('--random_seed', default=1234, type=int)
parser.add_argument('--epoch', default=20001, type=int)
parser.add_argument('--N_BC_train', default=10000,type=int) # train set. for inden, 2000. for inference 100
parser.add_argument('--N_IC_train', default=10000,type=int) # train set. for inden, 2000. for inference 100
parser.add_argument('--N_f_train', default=1000000,type=int) # train set. 
parser.add_argument("--nnlayers", default=[4, 20, 20, 20, 20, 20, 20, 20, 20, 1])
parser.add_argument('--dxi', default=100,type=int) # 100 point per 2 \mu m
parser.add_argument('--deta', default=100,type=int) # 100 point per 2 \mu m
parser.add_argument('--dz', default=40,type=int) # 40 point per 2 \mu m
parser.add_argument('--dt', default=5,type=int) # 1 point per 5 \mu s
parser.add_argument('--radius', default=1,type=float) 
parser.add_argument('--demiheight', default=50,type=float)
parser.add_argument('--Delta', default=10000,type=int)
parser.add_argument('--delta', default=5000,type=int)
parser.add_argument('--diffusivity', default=2e-3,type=float)
parser.add_argument('--bvalue', default=500,type=float)


# parser.add_argument('--input_dim', default=4, type=int) # input dimension = spatial + temporal
# parser.add_argument('--output_dim', default=1, type=int) # output dimension = 1

# parser.add_argument('--lr_pretrain', default=3e-4, type=float) # not sure whether we need to pretrain
# parser.add_argument('--epoch_pretrain', default=30001, type=int)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINNBTPDENN(nn.Module):
    def __init__(self,layerslist,diffusivity,delta,Delta,bvalue,dir):
        super(PINNBTPDENN, self).__init__()
        #input dim: 2 variables, spatial x and temporal t
        #output dim: 1. u    
        # [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
        'loss function'
        self.loss_function = nn.MSELoss(reduction ='mean')
        # self.loss_function = nn.L1Loss(reduction ='mean')

        self.diffusivity = diffusivity
        self.delta = delta
        self.Delta = Delta
        self.bvalue = bvalue
        self.qvalue = self.bvalue / (self.delta**2 * (self.Delta - self.delta / 3) )
        self.dir = dir

        # self.flatten = nn.Flatten()
        modules = []
        for i in range(0,len(layerslist)-1):
            nnlayer = nn.Linear(layerslist[i], layerslist[i+1]).to(torch.cfloat)
            nn.init.xavier_normal_(nnlayer.weight.data)
            nn.init.zeros_(nnlayer.bias.data)
            modules.append(nnlayer)
            if i != len(layerslist)-2:
                modules.append(nn.Tanh())

        self.layers = nn.Sequential(*modules)
        # alternative choice is nn.init.normal_ or nn.init.kaiming_uniform_ kaiming_normal_

        self.iter = 0

    def forward(self, X):
        # X is stack of x and t
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)

        H = self.layers(X.cfloat())
        return H

    def timeprofile(self, t):
        time_profile = torch.lt(t,self.delta).int() - torch.ge(t,self.Delta).int()
        return time_profile.cfloat()

    def loss_IC(self,x,y):          
        return self.loss_function(self.forward(x).real, y) + self.loss_function(self.forward(x).imag, torch.zeros_like(y))

    def loss_BC(self,x,y,dxieta_xy,x_circle):    
        # xi = x[:,[0]]
        # eta = x[:,[1]]             
        g = x.clone().cfloat()              
        g.requires_grad = True
        u = self.forward(g)
                
        u_1order = autograd.grad(u,g,torch.ones_like(u), retain_graph=True, create_graph=True)[0]

        dxidx,dxidy,detadx,detady = dxieta_xy

        u_xi = u_1order[:,[0]]
        u_eta = u_1order[:,[1]]

        u_normal = x_circle[:,[0]]*(u_xi*dxidx + u_eta*detadx) + x_circle[:,[1]]*(u_xi*dxidy + u_eta*detady)
        loss = self.loss_function(u_normal.real,y) + self.loss_function(u_normal.imag,y)
        return loss

    def loss_BC_top_bottom(self,x,y):
        # xi = x[:,[0]]
        # eta = x[:,[1]]
        # z = x[:,[2]]
        # t = x[:,[3]]              
        g = x.clone().cfloat()              
        g.requires_grad = True
        u = self.forward(g)       
        u_1order = autograd.grad(u,g,torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_zz = torch.sign(x[:,[2]]) * u_1order[:,[2]]
        loss = self.loss_function(u_zz.real,y) + self.loss_function(u_zz.imag,y)
        return loss
    
    def loss_PDE(self, x,y,dxieta_xy,x_circle):
        # xi = x[:,[0]]
        # eta = x[:,[1]]
        # z = x[:,[2]]
        # t = x[:,[3]]              
        g = x.clone().cfloat()              
        g.requires_grad = True
        u = self.forward(g)
                
        u_1order = autograd.grad(u,g,torch.ones_like(u), retain_graph=True, create_graph=True)[0]     
        u_xi2order = autograd.grad(u_1order[:,[0]],g,torch.ones_like(u_1order[:,[0]]), create_graph=True)[0]
        u_eta2order = autograd.grad(u_1order[:,[1]],g,torch.ones_like(u_1order[:,[1]]), create_graph=True)[0]
        u_z2order = autograd.grad(u_1order[:,[2]],g,torch.ones_like(u_1order[:,[2]]), create_graph=True)[0]

        
        u_t = u_1order[:,[3]]
        u_xixi = u_xi2order[:,[0]]
        u_xieta = u_xi2order[:,[1]]
        u_etaeta = u_eta2order[:,[1]]
        u_zz = u_z2order[:,[2]]

        dxidx,dxidy,detadx,detady = dxieta_xy
 

        f = u_t - self.diffusivity * ( u_zz + 
                                    dxidx*dxidx*u_xixi + 
                                    2*dxidx*detadx*u_xieta +  
                                    detadx*detadx*u_etaeta + 
                                    dxidy*dxidy*u_xixi + 
                                    2*dxidy*detady*u_xieta + 
                                    detady*detady*u_etaeta) + \
                                    torch.tensor(self.qvalue*1j).to(device)*self.timeprofile(x[:,[3]])*(x_circle[:,[2]]) * u
        
        loss_f = self.loss_function(f.real,y) + self.loss_function(f.imag,y)
                
        return loss_f
    
    def loss(self,x_IC,y_IC,x_BC,zero_BC,dxieta_xy_BC,x_circle_BC,x_BC_tb,zero_BC_tb,x_f,f_zero,dxieta_xy_f,x_circle_f):

        loss_IC = self.loss_IC(x_IC,y_IC)
        loss_u_BC = self.loss_BC(x_BC,zero_BC,dxieta_xy_BC,x_circle_BC)
        loss_u_BC_tb = self.loss_BC_top_bottom(x_BC_tb,zero_BC_tb) # top bottom
        loss_f = self.loss_PDE(x_f,f_zero,dxieta_xy_f,x_circle_f)
        
        loss_val = loss_IC + loss_f + loss_u_BC + loss_u_BC_tb
        
        return loss_val

    
    def predict(self, X):
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)
        u_pred = self(X).cpu().detach().numpy().squeeze()
        # u_pred = np.reshape(u_pred,(256,100),order='F')
        return u_pred

    def test(self, X_test, u):
                
        u_pred = self.forward(X_test)
        
        error_vec = torch.linalg.norm((u-u_pred),2)/torch.linalg.norm(u,2)        # Relative L2 Norm of the error (Vector)
        
        u_pred = u_pred.cpu().detach().numpy()
        
        u_pred = np.reshape(u_pred,(256,100),order='F')
                
        return error_vec, u_pred