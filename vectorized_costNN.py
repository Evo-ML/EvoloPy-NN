# -*- coding: utf-8 -*-
"""
Created on Fri May 27 12:03:15 2016

@author: hossam
"""
import numpy as np
import neurolab as nl
import time
#import warnings
#warnings.filterwarnings("ignore") 


def costNN(x,inputs,outputs,net):
    
    trainInput=inputs
        
    trainOutput=outputs
    
    numInputs=np.shape(trainInput)[1] #number of inputs
    
    
    #number of hidden neurons
    HiddenNeurons = net.layers[0].np['b'][:].shape[0]

    popSize = len(x)
       
    ######################################
    
    split1=HiddenNeurons*numInputs
    split2=split1+HiddenNeurons
    split3=split2+HiddenNeurons

    # input_w = 3X8 (HiddenNeurons*numInputs) 
    input_w =x[:, 0:split1].reshape(popSize, HiddenNeurons,numInputs)
    
    # layer_w = 1 X 3 (HiddenNeurons)
    layer_w=x[:, split1:split2].reshape(popSize, 1,HiddenNeurons)
 
    # input_bias = hiddenNeurons
    input_bias=x[:, split2:split3].reshape(popSize, 1,HiddenNeurons)
    #input_bias = np.array([0.4747,-1.2475,-1.2470])

    # bias_2 = 1
    bias_2 =x[:, split3:split3+1]

    nets = np.array([net] * popSize)

    nets = np.array(list(map(updateLayers, nets, input_w, layer_w, input_bias, bias_2)))
    '''
    net.layers[0].np['w'][:] = input_w
    net.layers[1].np['w'][:] = layer_w
    net.layers[0].np['b'][:] = input_bias
    net.layers[1].np['w'][:] = bias_2
    '''

    pred = np.array([net.sim(trainInput).reshape(len(trainOutput)) for net in nets])
    #pred=net.sim(trainInput).reshape(len(trainOutput))
    trainOutputs = np.array([trainOutput] * popSize)
    mse = ((pred - trainOutputs) ** 2).mean(axis=1)
    
    return mse

def updateLayers(net, input_w, layer_w, input_bias, bias_2):
    newNet = net.copy()
    newNet.layers[0].np['w'][:] = input_w
    newNet.layers[1].np['w'][:] = layer_w
    newNet.layers[0].np['b'][:] = input_bias
    newNet.layers[1].np['b'][:] = bias_2
    return newNet
    