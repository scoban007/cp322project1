from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import numpy as np
import math
import logisticRegression

# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(iris.metadata) 
  
# variable information 
print(iris.variables)

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
    totalAttr += xArray[i][0]
    totalAttr += xArray[i][1]
    totalAttr += xArray[i][2] / 6
    totalAttr += xArray[i][3]
    xFeature.append(totalAttr)
for j in range(len(yArray)):
    if yArray[j] == "Iris-setosa":
        yList1.append(1)
        yList2.append(0)
        yList3.append(0)
    if yArray[j] == "Iris-versicolor":
        yList1.append(0)
        yList2.append(1)
        yList3.append(0)
    if yArray[j] == "Iris-virginica":
        yList1.append(0)
        yList2.append(0)
        yList3.append(1)

learningRate = 0.001
iterations = 500

w, b = logisticRegression.fit(xFeature, yList1, learningRate, iterations)
print("\nAverage accuracy of predicting 'Iris Setosa' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList1, learningRate, iterations)))

w, b = logisticRegression.fit(xFeature, yList2, learningRate, iterations)
print("\nAverage accuracy of predicting 'Iris Versicolour' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList2, learningRate, iterations)))

w, b = logisticRegression.fit(xFeature, yList3, learningRate, iterations)
print("\nAverage accuracy of predicting 'Iris Virginica' is {:.2f} % with k-fold\n".format(logisticRegression.kfold(5, w, b, xFeature, yList3, learningRate, iterations)))