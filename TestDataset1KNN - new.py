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
xList = []

for i in xArray:
    xList.append(i)

xTest = xArray[0:20] #defining xTest 
#  xArray = xArray[5:10] #defining xArray, making it so that it contains different values than from xTest

for j in yArray: #making yList an int based list
    if j == "g":
        yList.append(1)
    else:
        yList.append(0)

sd = KNN.predicts(xArray, yList, xTest, 3) #pick k.
print(sd)
temp_sd = KNN.temp_predicts(xArray, yList, xTest, 3) #needed for evaluating accuracy, this list contains the targets of K  closest neighbours to x
#print(temp_sd)
print(KNN.evaluate_acc(temp_sd, yList[0:3])) #get accuracy
print("\nAverage accuracy is {:.2f} % with k-fold".format(KNN.kfold(xList, yList, 21, 3))) 
