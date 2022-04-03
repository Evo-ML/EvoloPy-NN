import optimizer as opt

# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizer=["PSO", "MVO", "GWO"]

#datasets=["flame","glass","iris","wdbc","Vertebral2"]
datasets=["flame","glass"]
        
# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=2

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 50, "Iterations": 5}

opt.run(optimizer, datasets, NumOfRuns, params)
                  
         
        
