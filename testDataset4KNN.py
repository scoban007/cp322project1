from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import math
from KNN import KNN
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 

# metadata 
#print(iris.metadata) 
  
# variable information 
#print(iris.variables) 
xArray = np.array(x)
yArray = np.array(y)
yList = []
xTest = []

xTest = xArray[0:5]


for j in yArray:
    if j == 'Iris-setosa':
        yList.append(1)
    elif j == 'Iris-versicolor':
        yList.append(2)
    else: 
        yList.append(3)


sd = KNN.predicts(xArray, yList, xTest, 5)
temp_sd = KNN.temp_predicts(xArray, yList, xTest, 5)
