# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:44:52 2016

@author: hossam
"""
import PSO as pso
import MVO as mvo
import GWO as gwo
import MFO as mfo
import CS as cs
import BAT as bat
import csv
import numpy
import time
import neurolab as nl
import costNN
import evaluateNetClassifier as evalNet
import solution

def selector(algo,func_details,popSize,Iter,trainDataset,testDataset):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    
    DatasetSplitRatio=2/3
    
    dataTrain="datasets/"+trainDataset
    dataTest="datasets/"+testDataset
    
    Dataset_train=numpy.loadtxt(open(dataTrain,"rb"),delimiter=",",skiprows=0)
    Dataset_test=numpy.loadtxt(open(dataTest,"rb"),delimiter=",",skiprows=0)
      
    
    numRowsTrain=numpy.shape(Dataset_train)[0]    # number of instances in the train dataset
    numInputsTrain=numpy.shape(Dataset_train)[1]-1 #number of features in the train dataset

    numRowsTest=numpy.shape(Dataset_test)[0]    # number of instances in the test dataset
    
    numInputsTest=numpy.shape(Dataset_test)[1]-1 #number of features in the test dataset
 

    trainInput=Dataset_train[0:numRowsTrain,0:-1]
    trainOutput=Dataset_train[0:numRowsTrain,-1]
    
    testInput=Dataset_test[0:numRowsTest,0:-1]
    testOutput=Dataset_test[0:numRowsTest,-1]
    
    #number of hidden neurons
    HiddenNeurons = numInputsTrain*2+1
    net = nl.net.newff([[0, 1]]*numInputsTrain, [HiddenNeurons, 1])
    
    dim=(numInputsTrain*HiddenNeurons)+(2*HiddenNeurons)+1;
    
    if(algo==0):
        x=pso.PSO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==1):
        x=mvo.MVO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==2):
        x=gwo.GWO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==3):
        x=mfo.MFO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==4):
        x=cs.CS(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==5):
        x=bat.BAT(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    

    # Evaluate MLP classification model based on the training set
    trainClassification_results=evalNet.evaluateNetClassifier(x,trainInput,trainOutput,net)
    x.trainAcc=trainClassification_results[0]
    x.trainTP=trainClassification_results[1]
    x.trainFN=trainClassification_results[2]
    x.trainFP=trainClassification_results[3]
    x.trainTN=trainClassification_results[4]
   
    # Evaluate MLP classification model based on the testing set   
    testClassification_results=evalNet.evaluateNetClassifier(x,testInput,testOutput,net)
    x.testAcc=testClassification_results[0]
    x.testTP=testClassification_results[1]
    x.testFN=testClassification_results[2]
    x.testFP=testClassification_results[3]
    x.testTN=testClassification_results[4] 
    
    
    return x
    
#####################################################################    
