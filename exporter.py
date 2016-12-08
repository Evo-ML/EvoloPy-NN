# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:13:19 2016

@author: hossam
"""

import csv
import matplotlib 
import math
import numpy as np

#def export(filename):

seperator = "Convg"
convergence=[]

Found = False
indexConvg=0

with open('convergence.csv', 'rt') as f:
     reader = csv.reader(f, delimiter=',') # good point by @paco
     for row in reader:
          index=0
          for field in row:
              index=index+1
              if field == seperator: 
                  Found=True
                  print ("is in file")
                  convergence.append (row[index:len(row)])
                  continue
#t=
t=np.array(convergence).astype(np.float)       
convgAvg = np.mean(t,axis=0)
     
if (Found== False):
    print("Convergence data can't be found in the specified file")
else:
    matplotlib.pyplot.plot( convgAvg, 'b-')
    #matplotlib.pyplot.title('A tale of 2 subplots')
    matplotlib.pyplot.ylabel('Average best fitness')
    matplotlib.pyplot.xlabel('Iteration')
    matplotlib.pyplot.grid(b=True)