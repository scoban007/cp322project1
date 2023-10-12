from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
from logisticRegression import fit

# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 
print(y)
print(x)
# metadata 
# print(ionosphere.metadata) 
  
# variable information 
# print(ionosphere.variables)

xArray = np.array(x)
yArray = np.array(y)
xFeature = []
yList = []

for i in xArray:
    totalAttr = 0
    for p in range(1, 6):   #select first 5 features to include
        totalAttr += i[p]
    xFeature.append(totalAttr)
    #xFeature.append(i[3])
for j in yArray:
    if j == "g":
        yList.append(1)
    else:
        yList.append(0)

w, b = fit(xFeature, yList, 0.01, 600)