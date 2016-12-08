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
import benchmarks
import csv
import numpy
import time
import neurolab as nl
import costNN
import evaluateNetClassifier as evalNet
import solution

def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    
    DatasetSplitRatio=2/3
    
 
    Dataset=numpy.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=0)
    
    numRows=numpy.shape(Dataset)[0]    # number of instances in the dataset
    numInputs=numpy.shape(Dataset)[1]-1 #number of features in the dataset

    splitPoint= int(DatasetSplitRatio*numRows)

    trainInput=Dataset[0:splitPoint,0:-1]
    trainOutput=Dataset[0:splitPoint,-1]
    
    testInput=Dataset[splitPoint:,0:-1]
    testOutput=Dataset[splitPoint:,-1]
   
    
    #number of hidden neurons
    HiddenNeurons = numInputs*2+1
    net = nl.net.newff([[0, 1]]*numInputs, [HiddenNeurons, 1])
    
    dim=(numInputs*HiddenNeurons)+(2*HiddenNeurons)+1;
    
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
