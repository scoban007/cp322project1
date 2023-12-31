from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import math
import logisticRegression

# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
x = wine.data.features 
y = wine.data.targets 
  
# metadata 
print(wine.metadata) 
  
# variable information 
print(wine.variables)

print(x)
print(y)

xArray = np.array(x)
yArray = np.array(y)
xFeature = []
yList1 = []
yList2 = []
yList3 = []


for i in range(len(xArray)):
    totalAttr = 0
    totalAttr += xArray[i][1] / 10
    totalAttr += xArray[i][3] / 25
    totalAttr += xArray[i][9]
    totalAttr += xArray[i][10]
    totalAttr += xArray[i][11]
    xFeature.append(totalAttr)

for j in range(len(yArray)):
    if yArray[j] == 1:
        yList1.append(1)
        yList2.append(0)
        yList3.append(0)
    if yArray[j] == 2:
        yList1.append(0)
        yList2.append(1)
        yList3.append(0)
    if yArray[j] == 3:
        yList1.append(0)
        yList2.append(0)
        yList3.append(1)

    
learningRate = 0.001
iterations = 500

w, b = logisticRegression.fit(xFeature, yList1, learningRate, iterations)
print("\nAverage accuracy of predicting 'Wine 1' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList1, learningRate, iterations)))

w, b = logisticRegression.fit(xFeature, yList2, learningRate, iterations)
print("\nAverage accuracy of predicting 'Wine 2' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList2, learningRate, iterations)))

w, b = logisticRegression.fit(xFeature, yList3, learningRate, iterations)
print("\nAverage accuracy of predicting 'Wine 3' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList3, learningRate, iterations)))