from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import math
from KNN import KNN
  
# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
x = wine.data.features 
y = wine.data.targets 
xArray = np.array(x)
yArray = np.array(y)
yList = []
xTest = []

xTest = xArray[55]
print(yArray)

for j in yArray:
    if j == 1:
        yList.append(1)
    elif j == 2:
        yList.append(2)
    else: 
        yList.append(3)


sd = KNN.predicts(xArray, yList, xTest, 5)

temp_sd = KNN.temp_predicts(xArray, yList, xTest, 5)

