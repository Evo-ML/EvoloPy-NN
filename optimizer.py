# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:44:52 2016

@author: hossam
"""
import csv
import numpy
import time
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

def run(optimizer, datasets, NumOfRuns, params):

    #Export results ?
    Export=True

    #ExportToFile="YourResultsAreHere.csv"
    #Automaticly generated file name by date and time
    ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

    # Check if it works at least once
    Flag=False

    # CSV Header for for the cinvergence 
    CnvgHeader=[]
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    for l in range(0,Iterations):
        CnvgHeader.append("Iter"+str(l+1))
    
    for j in range (0, len(datasets)):        # specfiy the number of the datasets
        for i in range (0, len(optimizer)):    
            for k in range (0,NumOfRuns):
                
                func_details=["costNN",-1,1]

                dataset="datasets/"+datasets[j]+".csv"

                    
                dataset_values=numpy.loadtxt(open(dataset,"rb"),delimiter=",")
             
                X = numpy.array(dataset_values)[:,:-1]
                y = numpy.array(dataset_values)[:,-1]

                trainInput, testInput, trainOutput, testOutput = train_test_split(X, y, test_size=0.33, random_state=42)
                
                numFeatures=numpy.shape(trainInput)[1]#number of features in the train dataset

                #number of hidden neurons
                HiddenNeurons = numFeatures*2+1
                net = nl.net.newff([[0, 1]]*numFeatures, [HiddenNeurons, 1])
                
                dim=(numFeatures*HiddenNeurons)+(2*HiddenNeurons)+1;
                    
                x = selector(optimizer[i], func_details, dim, PopulationSize, Iterations, trainInput,trainOutput,net)

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
                
                
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime","trainAcc", "trainTP","trainFN","trainFP","trainTN", "testAcc", "testTP","testFN","testFP","testTN"],CnvgHeader])
                            writer.writerow(header)
                        a=numpy.concatenate([[x.optimizer,datasets[j],x.objfname,k+1,x.startTime,x.endTime,x.executionTime,x.trainAcc, x.trainTP,x.trainFN,x.trainFP,x.trainTN, x.testAcc, x.testTP,x.testFN,x.testFP,x.testTN],x.convergence])
                        writer.writerow(a)
                    out.close()
                Flag=True # at least one experiment


    if (Flag==False): # Faild to run at least one experiment
        print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        

def selector(algo, func_details, dim, popSize, Iter, trainInput,trainOutput,net):


    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]                  

    
    if(algo=="PSO"):
        x=pso.PSO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="MVO"):
        x=mvo.MVO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="GWO"):
        x=gwo.GWO( getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="MFO"):
        x=mfo.MFO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="CS"):
        x=cs.CS(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="BAT"):
        x=bat.BAT(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="WOA"):
        x=woa.WOA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="FFA"):
        x=ffa.FFA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="SSA"):
        x=ssa.SSA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="GA"):
        x=ga.GA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="HHO"):
        x=hho.HHO(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="SCA"):
        x=sca.SCA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="JAYA"):
        x=jaya.JAYA(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    elif(algo=="DE"):
        x=de.DE(getattr(costNN, function_name),lb,ub,dim,popSize,Iter,trainInput,trainOutput,net)
    else:
        return None
    return x



#####################################################################    
