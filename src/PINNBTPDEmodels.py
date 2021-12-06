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
parser.add_argument('--N_BC_train', default=1000,type=int) # train set. for inden, 2000. for inference 100
parser.add_argument('--N_IC_train', default=1000,type=int) # train set. for inden, 2000. for inference 100
parser.add_argument('--N_f_train', default=10000,type=int) # train set. 
parser.add_argument("--nnlayers", default=[4, 100, 100, 100, 100, 2])
parser.add_argument('--dxi', default=100,type=int) # 100 point per 2 \mu m
parser.add_argument('--deta', default=100,type=int) # 100 point per 2 \mu m
parser.add_argument('--dz', default=40,type=int) # 40 point per 2 \mu m
parser.add_argument('--dt', default=5,type=int) # 1 point per 5 \mu s
parser.add_argument('--radius', default=1,type=float) 
parser.add_argument('--demiheight', default=1,type=float)
parser.add_argument('--Delta', default=10000,type=int)
parser.add_argument('--delta', default=5000,type=int)
parser.add_argument('--diffusivity', default=2e-3,type=float)
parser.add_argument('--bvalue', default=50,type=float)


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINNBTPDENN(nn.Module):
    def __init__(self,layerslist,diffusivity,delta,Delta,bvalue,dir):
        super(PINNBTPDENN, self).__init__()
        #input dim: 2 variables, spatial x and temporal t
        #output dim: 1. u    
        # [4, 100, 100, 100, 100, 100, 100, 100, 100, 2]
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
            nnlayer = nn.Linear(layerslist[i], layerslist[i+1]).to(torch.float)
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

        H = self.layers(X.float())

        Hreal = H[:,[0]]
        Himag = H[:,[1]]
        return Hreal, Himag

    def timeprofile(self, t):
        time_profile = torch.lt(t,self.delta).int() - torch.ge(t,self.Delta).int()
        return time_profile.float()

    def loss_IC(self,x,y):
        outputreal, outputimag = self.forward(x)        
        return self.loss_function(outputreal, y) + self.loss_function(outputimag, torch.zeros_like(y))

    def loss_BC(self,x,y,dxieta_xy,x_circle):    
        # xi = x[:,[0]]
        # eta = x[:,[1]]             
        g = x.clone().float()              
        g.requires_grad = True
        ureal, uimag = self.forward(g)
                
        ureal_1order = autograd.grad(ureal,g,torch.ones_like(ureal), retain_graph=True, create_graph=True)[0]

        uimag_1order = autograd.grad(uimag,g,torch.ones_like(uimag), retain_graph=True, create_graph=True)[0]

        dxidx,dxidy,detadx,detady = dxieta_xy

        ureal_xi = ureal_1order[:,[0]]
        ureal_eta = ureal_1order[:,[1]]

        uimag_xi = uimag_1order[:,[0]]
        uimag_eta = uimag_1order[:,[1]]

        ureal_normal = self.diffusivity*( x_circle[:,[0]]*(ureal_xi*dxidx + ureal_eta*detadx) + x_circle[:,[1]]*(ureal_xi*dxidy + ureal_eta*detady) )
        uimag_normal = self.diffusivity*( x_circle[:,[0]]*(uimag_xi*dxidx + uimag_eta*detadx) + x_circle[:,[1]]*(uimag_xi*dxidy + uimag_eta*detady) )

        loss = self.loss_function(ureal_normal,y) + self.loss_function(uimag_normal,y)
        return loss

    def loss_BC_top_bottom(self,x,y):
        # xi = x[:,[0]]
        # eta = x[:,[1]]
        # z = x[:,[2]]
        # t = x[:,[3]]              
        g = x.clone().float()              
        g.requires_grad = True
        ureal, uimag = self.forward(g)       
        ureal_1order = autograd.grad(ureal,g,torch.ones_like(ureal), retain_graph=True, create_graph=True)[0]

        uimag_1order = autograd.grad(uimag,g,torch.ones_like(uimag), retain_graph=True, create_graph=True)[0]
        ureal_zz = torch.sign(x[:,[2]]) * ureal_1order[:,[2]]
        uimag_zz = torch.sign(x[:,[2]]) * uimag_1order[:,[2]]

        loss = self.loss_function(ureal_zz,y) + self.loss_function(uimag_zz,y)
        return loss
    
    def loss_PDE(self, x,y,dxieta_xy,x_circle):
        # xi = x[:,[0]]
        # eta = x[:,[1]]
        # z = x[:,[2]]
        # t = x[:,[3]]              
        g = x.clone().float()              
        g.requires_grad = True
        ureal,uimag = self.forward(g)
                
        ureal_1order = autograd.grad(ureal,g,torch.ones_like(ureal), retain_graph=True, create_graph=True)[0]     
        ureal_xi2order = autograd.grad(ureal_1order[:,[0]],g,torch.ones_like(ureal_1order[:,[0]]), create_graph=True)[0]
        ureal_eta2order = autograd.grad(ureal_1order[:,[1]],g,torch.ones_like(ureal_1order[:,[1]]), create_graph=True)[0]
        ureal_z2order = autograd.grad(ureal_1order[:,[2]],g,torch.ones_like(ureal_1order[:,[2]]), create_graph=True)[0]

        uimag_1order = autograd.grad(uimag,g,torch.ones_like(uimag), retain_graph=True, create_graph=True)[0]     
        uimag_xi2order = autograd.grad(uimag_1order[:,[0]],g,torch.ones_like(uimag_1order[:,[0]]), create_graph=True)[0]
        uimag_eta2order = autograd.grad(uimag_1order[:,[1]],g,torch.ones_like(uimag_1order[:,[1]]), create_graph=True)[0]
        uimag_z2order = autograd.grad(uimag_1order[:,[2]],g,torch.ones_like(uimag_1order[:,[2]]), create_graph=True)[0]

        
        ureal_t = ureal_1order[:,[3]]
        ureal_xixi = ureal_xi2order[:,[0]]
        ureal_xieta = ureal_xi2order[:,[1]]
        ureal_etaeta = ureal_eta2order[:,[1]]
        ureal_zz = ureal_z2order[:,[2]]

        uimag_t = uimag_1order[:,[3]]
        uimag_xixi = uimag_xi2order[:,[0]]
        uimag_xieta = uimag_xi2order[:,[1]]
        uimag_etaeta = uimag_eta2order[:,[1]]
        uimag_zz = uimag_z2order[:,[2]]

        dxidx,dxidy,detadx,detady = dxieta_xy
 
        ftx = self.timeprofile(x[:,[3]])*(x_circle[:,[2]])

        freal = ureal_t - self.diffusivity * ( ureal_zz + 
                                    dxidx*dxidx*ureal_xixi + 
                                    2*dxidx*detadx*ureal_xieta +  
                                    detadx*detadx*ureal_etaeta + 
                                    dxidy*dxidy*ureal_xixi + 
                                    2*dxidy*detady*ureal_xieta + 
                                    detady*detady*ureal_etaeta) - \
                                     self.qvalue*ftx* uimag

        fimag = uimag_t - self.diffusivity * ( uimag_zz + 
                                    dxidx*dxidx*uimag_xixi + 
                                    2*dxidx*detadx*uimag_xieta +  
                                    detadx*detadx*uimag_etaeta + 
                                    dxidy*dxidy*uimag_xixi + 
                                    2*dxidy*detady*uimag_xieta + 
                                    detady*detady*uimag_etaeta) + \
                                    self.qvalue*ftx * ureal
        
        loss_f = self.loss_function(freal,y) + self.loss_function(fimag,y)
                
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
        ureal_pred, uimag_pred  = self(X)
        ureal_pred = ureal_pred.cpu().detach().numpy().squeeze()
        uimag_pred = uimag_pred.cpu().detach().numpy().squeeze()
        # u_pred = np.reshape(u_pred,(256,100),order='F')
        return ureal_pred, uimag_pred

    def test(self, X_test, ureal, uimag):
                
        ureal_pred, uimag_pred = self.forward(X_test)
        
        error_vec = torch.linalg.norm((ureal-ureal_pred),2)/torch.linalg.norm(ureal,2) + torch.linalg.norm((uimag-uimag_pred),2)/torch.linalg.norm(uimag,2)        # Relative L2 Norm of the error (Vector)
        
        ureal_pred = ureal_pred.cpu().detach().numpy()
        uimag_pred = uimag_pred.cpu().detach().numpy()
        
        # u_pred = np.reshape(u_pred,(256,100),order='F')
                
        return error_vec, ureal_pred, uimag_pred