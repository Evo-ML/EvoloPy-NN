import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

def run(results_directory, optimizer, dataset_List, ev_measures, Iterations):
    plt.ioff()
    
    fileResultsDetailsData = pd.read_csv(results_directory + '/experiment.csv')
    for d in range(len(dataset_List)):        
        dataset_filename = dataset_List[d] + '.csv' 
        for z in range (0, len(ev_measures)):
            
            #Box Plot
            data = []      
                
            for i in range(len(optimizer)): 
                optimizer_name = optimizer[i]
                
                detailedData = fileResultsDetailsData[(fileResultsDetailsData["Dataset"] == dataset_List[d]) & (fileResultsDetailsData["Optimizer"] == optimizer_name)]
                detailedData = detailedData[ev_measures[z]]
                detailedData = np.array(detailedData).T.tolist()
                data.append(detailedData)

            #, notch=True
            box=plt.boxplot(data,patch_artist=True,labels=optimizer)
            

            colors = ['#5c9eb7','#f77199', '#cf81d2','#4a5e6a','#f45b18',
            '#ffbd35','#6ba5a1','#fcd1a1','#c3ffc1','#68549d',
            '#1c8c44','#a44c40','#404636']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
             
            plt.legend(handles= box['boxes'], labels=optimizer, 
                loc="upper right", bbox_to_anchor=(1.2,1.02))
            fig_name = results_directory + "/boxplot-" + dataset_List[d] + "-" + ev_measures[z] + ".png"
            plt.savefig(fig_name, bbox_inches='tight')
            plt.clf()
            #plt.show()
            


