# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:44:52 2016

@author: hossam
"""
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import csv
import numpy
import time
import neurolab as nl
import costNN
import evaluateNetClassifier as evalNet
import solution
from sklearn.model_selection import train_test_split

def selector(algo,func_details,popSize,Iter,dataset):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    
    DatasetSplitRatio=2/3
        
    dataset_values=numpy.loadtxt(open(dataset,"rb"),delimiter=",")
 
    X = numpy.array(dataset_values)[:,:-1]
    y = numpy.array(dataset_values)[:,-1]

    trainInput, testInput, trainOutput, testOutput = train_test_split(X, y, test_size=0.33, random_state=42)
    
    numFeatures=numpy.shape(trainInput)[1]#number of features in the train dataset

    #number of hidden neurons
    HiddenNeurons = numFeatures*2+1
    net = nl.net.newff([[0, 1]]*numFeatures, [HiddenNeurons, 1])
    
    dim=(numFeatures*HiddenNeurons)+(2*HiddenNeurons)+1;
    
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
    if(algo==6):
        x=woa.WOA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==7):
        x=ffa.FFA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==8):
        x=ssa.SSA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==9):
        x=ga.GA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==10):
        x=hho.HHO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==11):
        x=sca.SCA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==12):
        x=jaya.JAYA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    if(algo==13):
        x=de.DE(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
        

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
