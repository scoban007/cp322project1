from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
import logisticRegression

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

for i in range(250):    #pick first 250 entries from dataset
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

learningRate = 0.01
iterations = 500

w, b = logisticRegression.fit(xFeature, yList, learningRate, iterations)
print("\nAverage accuracy is {:.2f} % with k-fold".format(logisticRegression.kfold(5, w, b, xFeature, yList, learningRate, iterations)))
