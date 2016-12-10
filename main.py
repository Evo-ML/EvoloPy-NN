
import csv
import numpy
import time
import selector as slctr

# Select optimizers
PSO= True
MVO= True
GWO = True
MFO= True
CS= True
BAT=True



optimizer=[PSO, MVO, GWO, MFO, CS, BAT]
datasets=["BreastCancer", "Diabetes", "Liver", "Parkinsons", "Vertebral"]
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
NumOfRuns=2

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations= 5

#Export results ?
Export=True


#ExportToFile="YourResultsAreHere.csv"
#Automaticly generated file name by date and time
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

trainDataset="breastTrain.csv"
testDataset="breastTest.csv"
for j in range (0, len(datasets)):        # specfiy the number of the datasets
    for i in range (0, len(optimizer)):
    
        if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                
                func_details=["costNN",-1,1]
                trainDataset=datasets[j]+"Train.csv"
                testDataset=datasets[j]+"Test.csv"
                x=slctr.selector(i,func_details,PopulationSize,Iterations,trainDataset,testDataset)
                  
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
        
        
