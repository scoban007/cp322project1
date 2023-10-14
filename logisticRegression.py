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
    #logReg = 1/(1 + np.exp(-(xrange)))

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
    
        cost = 0
        for j in range(len(xFeature)):
            cost += loss(temp_w, temp_b, xFeature[j], yList[j])
        cost = -(cost/len(xFeature))
        
        params.append([cost, temp_w, temp_b])


    min = params[0][0]
    for vals in params:
        if vals[0] < min:
            min = vals[0]
            bestParams = vals

    return bestParams[1], bestParams[2]

def predict(w, b, xFeature):
    # calculates probability of the positive class 
    probability = 1 / (1 + np.exp(-(w * xFeature + b)))

    # return class prediction 0 or 1 based on a threshold
    if probability >= 0.5:
        return 1
    else:
        return 0

def evaluate_acc(w, b, xList, yList, sampleSize):

    accuracy = 0

    for i in range(sampleSize):
        if predict(w, b, xList[i]) == yList[i]:
            accuracy += 1
    
    return 100 * accuracy / sampleSize

def kfold(k, w, b, xList, yList):

    kSize = len(xList) // k
    avgAcc = 0

    for i in range(k):

        startIndex = i*kSize
        endIndex = (i+1)*kSize

        testingSet_x = xList[startIndex:endIndex]
        testingSet_y = yList[startIndex:endIndex]
        trainingSet_x = xList[0:startIndex] + xList[endIndex:]
        trainingSet_y = yList[0:startIndex] + yList[endIndex:]

        w,b = fit(trainingSet_x, trainingSet_y, 0.01, 600)
        print("Accuracy for fold #{} is {:.2f} %".format(i+1, evaluate_acc(w, b, testingSet_x, testingSet_y, len(testingSet_x))))

        avgAcc += evaluate_acc(w, b, testingSet_x, testingSet_y, len(testingSet_x))

    return avgAcc / k
