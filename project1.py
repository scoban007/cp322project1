from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math

  
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
x = ionosphere.data.features 
y = ionosphere.data.targets 
# metadata 
# print(ionosphere.metadata) 
  
# variable information 
# print(ionosphere.variables)

def seq_W(w, xFeature, yBinary):
    return 1/2*(1/(1 + np.exp(-(w*xFeature)))-yBinary)**2

def seq_B(b, xFeature, yBinary):
    return 1/2*(1/(1 + np.exp(-(b)))-yBinary)**2

def loss_W(w, xFeature, yBinary):
    return yBinary * math.log(1/(1 + np.exp(-(w*xFeature)))) + (1 - yBinary)*math.log(1 - (1/(1 + (np.exp(-(w*xFeature))))))

def fit(x, y):

    xArray = np.array(x)
    yArray = np.array(y)
    xFeature = []
    yList = []
    print(xArray[3])
    
    for i in xArray:
        totalAttr = 0
        for p in range(1, 6):   #select features to include
            totalAttr += i[p]
        xFeature.append(totalAttr)
    for j in yArray:
        yList.append(j[0])
    
    #plt.scatter(xFeature, yList, color="red")   #data
    xrange = np.array(xFeature)
    logReg = 1/(1 + np.exp(-(xrange)))
    #plt.scatter(xrange, logReg, color="blue")   #logistic regression

    for w in range(0, 100):         #squared error cost for w
        seq = 0
        for j in range(len(xFeature)):
            if yList[j] == "g":
                yBin = 0
            else:
                yBin = 1
            seq += loss_W(0.05*w, xFeature[j], yBin)   #1/2*(1/(1 + np.exp(-(i*xFeature[j])))-yBin)**2
        seq = seq/len(xFeature)
        print("{} : {}".format(w, seq))
        plt.scatter(seq,0.05*w,color="green")
    
    # for i in range(0, 10):         #squared error cost for b
    #     seq = 0
    #     for j in range(len(xFeature)):
    #         if yList[j] == "g":
    #             yBin = 0
    #         else:
    #             yBin = 1
    #         seq += 1/2*(1/(1 + np.exp(-(i)))-yBin)**2
    #     seq = seq/len(xFeature)
    #     print("{} : {}".format(i, seq))
    #     plt.scatter(seq,i,color="orange")

    
    plt.show()

fit(x, y)