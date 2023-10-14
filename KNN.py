import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import math
import operator
import random
from collections import Counter
ionosphere = fetch_ucirepo(id=52) 
X = ionosphere.data.features 
y = ionosphere.data.targets


class KNN:

    def __init__(self):
        self.xArray = np.array(X)
        self.yArray = np.array(y)
        return
    
    def fit(self, k):
        self.k = k #we dont need much for fit

    def euclidean (p , q):
        return np.sqrt((np.sum(np.array(p) - np.array(q)) ** 2)) #euclidean for finding distance between every point in p and q.
    
    def predicts(X_train, y_train, X_test, k):
        temp_prediction = []
        temp_distances = []
        distances = []
        prediction = []
        h = 0
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                temp_distances.append([(KNN.euclidean(X_train[j], X_test[i])), j])
            temp_distances.sort()
        distances = temp_distances[:k]
        while h < len(distances):
            for x, y in distances:
                temp_prediction.append(y_train[y])
                h += 1
        prediction = Counter(temp_prediction).most_common(1)[0][0]
        return np.array(prediction)
    
    def temp_predicts(X_train, y_train, X_test, k):
        temp_prediction = []
        temp_distances = []
        distances = []
        prediction = []
        h = 0
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                temp_distances.append([(KNN.euclidean(X_train[j], X_test[i])), j])
            temp_distances.sort()
        distances = temp_distances[:k]
        while h < len(distances):
            for x, y in distances:
                temp_prediction.append(y_train[y])
                h += 1
        return temp_prediction
    
    def evaluate_acc(predictions, y_test):
        sumAll = 0
        for i in range(len(predictions)):
            if predictions[i] == y_test[i]:
                sumAll += 1            
        return sumAll/len(y_test)
    
    def kfold(xList, yList, ksize, k): #its not working :'DDDDDDDD
        kSize = len(xList) // ksize
        avgAcc = 0
        predictions = []

        for i in range(kSize - 1):

            startIndex = i*kSize
            endIndex = (i+1)*kSize
            print(endIndex)

            testingSet_x = xList[startIndex:endIndex]
            testingSet_y = yList[startIndex:endIndex]
            print(testingSet_y)

            predictions.append(KNN.predicts(xList, testingSet_y, testingSet_x, k))
            print(predictions)
            avgAcc += KNN.evaluate_acc(predictions, testingSet_y)

        return avgAcc / k
