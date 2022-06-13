import vectorized_optimizer as opt

# Select optimizers
# "SSA","PSO","GA","BAT","FFA","GWO","WOA","MVO","MFO","CS","HHO","SCA","JAYA","DE"
optimizer=["GWO", "MFO", "SCA", "SSA", "WOA"]#"GWO", "MFO", "SCA", "SSA", "WOA"
#optimizer=["GWO","MFO"]#"GWO", "MFO", "SCA", "SSA", "WOA"

datasets=["flame","glass","iris","wdbc","Vertebral2"]
#datasets=["flame"]
        
# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=2

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 50, "Iterations": 100}

opt.run(optimizer, datasets, NumOfRuns, params)
                  
         
        
