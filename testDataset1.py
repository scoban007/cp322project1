from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
import logisticRegression

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

w, b = logisticRegression.fit(xFeature, yList, 0.01, 600)

print("Accuracy is {:.2f} %\n".format(logisticRegression.evaluate_acc(w, b, xFeature, yList, 350)))
print("\nAverage accuracy is {:.2f} % with k-fold".format(logisticRegression.kfold(7, w, b, xFeature, yList)))

