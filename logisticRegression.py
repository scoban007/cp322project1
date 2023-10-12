from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import math

def logReg(w, xFeature, b):
    return 1/(1 + (np.exp(-(w * xFeature + b))))

def loss(w, b, xFeature, yBinary):
    return yBinary * math.log(logReg(w, xFeature, b)) + (1 - yBinary)*math.log(1 - logReg(w, xFeature, b))

def grad_desc_w(w, xFeature, b, yBin):
    return (logReg(w, xFeature, b) - yBin) * xFeature

def grad_desc_b(w, xFeature, b, yBin):
    return logReg(w, xFeature, b) - yBin

def fit(xFeature, yList, learningRate, iterations):
    
    params = []

    xrange = np.array(xFeature)
    logReg = 1/(1 + np.exp(-(xrange)))

    temp_w = 0
    temp_b = 0
    for w in range(0, iterations):    #find w and b
        grad_w = 0
        grad_b = 0

        for i in range(len(xFeature)):
            grad_w += grad_desc_w(temp_w, xFeature[i], temp_b, yList[i])
            grad_b += grad_desc_b(temp_w, xFeature[i], temp_b, yList[i])
        temp_w = temp_w - learningRate*grad_w/len(xFeature)
        temp_b = temp_b - learningRate*grad_b/len(xFeature)

        #print("J({}, {}): {}, {}".format(wScaled, b, temp_w, temp_b))
        #plt.scatter(xrange, 1/(1 + np.exp(-(temp_w*xrange+temp_b))))
    
        cost = 0
        for j in range(len(xFeature)):
            cost += loss(temp_w, temp_b, xFeature[j], yList[j])
        cost = -(cost/len(xFeature))
        
        params.append([cost, temp_w, temp_b])

        #print("J({}, {}): {}".format(temp_w, temp_b, cost))
        #print("{} : {}".format(wScaled, cost))
        plt.scatter(w, cost, color="green")
    plt.show()

    min = params[0][0]
    for vals in params:
        if vals[0] < min:
            min = vals[0]
            bestParams = vals

    print(bestParams)
    logReg = 1/(1 + np.exp(-(bestParams[1]*xrange + bestParams[2])))
    plt.scatter(xFeature, yList, color="red")
    plt.scatter(xrange, logReg, color="blue")
    #plt.scatter(xrange, 1/(1 + np.exp(-(1*xrange + 0))), color="green")
    plt.show()

    return bestParams[1], bestParams[2]
