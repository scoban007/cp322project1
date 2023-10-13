from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
from KNN import KNN

adult = fetch_ucirepo(id=2) 
x = adult.data.features 
y = adult.data.targets 

xArray = np.array(x)
yArray = np.array(y)
xFeature = []
yList = []
xTest = []


for i in range(250):
    xFeature.append(xArray[i][10]) #only considered capital gain/loss and hours per week
    xFeature.append(xArray[i][11])
    xFeature.append(xArray[i][12])

for i in range(5):
    xTest.append(xArray[i][10])
    xTest.append(xArray[i][11])
    xTest.append(xArray[i][12])

print(xTest)

for j in range(250):
    if yArray[j] == ">50K":
        yList.append(0)
    else:
        yList.append(1)

sd = KNN.predicts(xFeature, yList, xTest, 5)
#print(sd)
temp_sd = KNN.temp_predicts(xFeature, yList, xTest, 5)
#print(temp_sd)
#print(KNN.evaluate_acc(temp_sd, yList[:5]))
