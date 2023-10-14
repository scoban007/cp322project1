from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math
import logisticRegression

# fetch dataset 
wine = fetch_ucirepo(id=109) 
  
# data (as pandas dataframes) 
X = wine.data.features 
y = wine.data.targets 


# # metadata 
#print(wine.metadata) 
  
# # variable information 
# print(wine.variables) 

xArray = np.array(X)
yArray = np.array(y)
xFeature = []
yList = []


for i in xArray:
    totalAttr = 0
    for p in range(1, 5):   #select first 5 features to include
        totalAttr += i[p]
    
    xFeature.append(totalAttr)
    #xFeature.append(i[3])
for j in yArray:
    if j == "1":
        yList.append(1)
    else:
        yList.append(0)



w, b = logisticRegression.fit(xFeature, yList, 0.01, 600)

#keep getting out of bounds error, I'm not sure if it's the data not formatted right or whattttt PS im sorry 

print("Accuracy is {:.2f} %\n".format(logisticRegression.evaluate_acc(w, b, xFeature, yList, 350)))
print("\nAverage accuracy is {:.2f} % with k-fold".format(logisticRegression.kfold(7, w, b, xFeature, yList)))