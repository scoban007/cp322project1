from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
from KNN import KNN

ionosphere = fetch_ucirepo(id=52) 
x = ionosphere.data.features 
y = ionosphere.data.targets 

xArray = np.array(x)
yArray = np.array(y)
yList = []
xTest = []

xTest = xArray[0:5] #define k as 5

for j in yArray:
    if j == "g":
        yList.append(1)
    else:
        yList.append(0)

sd = KNN.predicts(xArray, yList, xTest, 5) 
#print(sd)
temp_sd = KNN.temp_predicts(xArray, yList, xTest, 5) #need all of the predictions until k to compare and get accuracy
#print(temp_sd)
#print(KNN.evaluate_acc(temp_sd, yList[:5]))
