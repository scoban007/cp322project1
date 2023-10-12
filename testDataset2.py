from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
from logisticRegression import fit

# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
x = adult.data.features 
y = adult.data.targets 
print(y)
print(x)
# metadata 
#print(adult.metadata) 
  
# variable information 
#print(adult.variables)

xArray = np.array(x)
yArray = np.array(y)
xFeature = []
yList = []

for i in range(250):
    totalAttr = 0
    totalAttr += xArray[i][0] / 75
    totalAttr += xArray[i][2] / 450000
    totalAttr += xArray[i][4] / 18
    totalAttr += xArray[i][12] / 100
    xFeature.append(totalAttr)
for j in range(250):
    if yArray[j] == ">50K":
        yList.append(0)
    else:
        yList.append(1)

print(len(xFeature))
bestParams = fit(xFeature, yList, 0.004, 400)