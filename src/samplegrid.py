# -*- coding: utf-8 -*-
import numpy as np

def samplegrid_v2(input,Nsample,axe,axevalue):
    outputlist = np.array([] ).reshape(Nsample,0)
    for ind_input in range( len(input) ):
        if ind_input == axe:
            tmp = np.ones( (Nsample,1) )*axevalue
        else:
            if ind_input==3 :
                input_tmp = input[ind_input]
                id_tmp = np.random.choice(input_tmp.size, Nsample)
                tmp = input_tmp[id_tmp]
            else:
                input_tmp = input[ind_input]
                id_tmp = np.random.choice(input_tmp.size-2, Nsample)
                tmp = input_tmp[id_tmp+1]
        outputlist = np.append(outputlist, tmp.reshape( (-1,1) ), axis = 1 )
    output = np.vstack(outputlist)
    return output

def samplegrid(input,Nsample,axe,axevalue):
    outputlist = np.array([] ).reshape(Nsample,0)
    for ind_input in range( len(input) ):
        if ind_input == axe:
            tmp = np.ones( (Nsample,1) )*axevalue
        else:
            input_tmp = input[ind_input]
            id_tmp = np.random.choice(input_tmp.size, Nsample)
            tmp = input_tmp[id_tmp]
        outputlist = np.append(outputlist, tmp.reshape( (-1,1) ), axis = 1 )
    output = np.vstack(outputlist)
    return output