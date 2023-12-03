import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import math
import operator
import random
from collections import Counter
ionosphere = fetch_ucirepo(id=2) 
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
        np.array(p)
        return np.sqrt((np.sum(np.array(p) - np.array(q)) ** 2)) #euclidean for finding distance between every point in p and q.
    
    def predicts(X_train, y_train, X_test, k):

        prediction = []
        for i in range(len(X_test)):    #iterate through each testing example
            temp_prediction = []
            temp_distances = []
            distances = []
            for j in range(len(X_train)):   #for each testing example, iterate through training data to find k nearest neighbours using euclidean dist
                temp_distances.append([(KNN.euclidean(X_train[j], X_test[i])), j])
            temp_distances.sort()   #sort by distance
            distances = temp_distances[:k]      #get first k distances
            print("\ndistances: {}".format(distances))
            for y in range(len(distances)):     #get first k nearest neighbours from training data
                temp_prediction.append(y_train[distances[y][1]])
            print("temp_prediction: {}".format(temp_prediction))
            prediction.append(Counter(temp_prediction).most_common(1)[0][0])    #append prediction using most common y value
            print("most common: {}".format(Counter(temp_prediction).most_common(1)[0][0]))
        return prediction   #return list of predictions; one prediction for each training example
    
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
        return 100 * sumAll/len(y_test)
    
    def kfold(xList, yList, ksize, k):
        kSize = len(xList) // ksize
        avgAcc = 0

        acc = []

        for i in range(ksize):
            predictions = []
            startIndex = i*kSize
            endIndex = (i+1)*kSize

            testingSet_x = xList[startIndex:endIndex]
            testingSet_y = yList[startIndex:endIndex]
            trainingSet_x = xList[0:startIndex] + xList[endIndex:]
            trainingSet_y = yList[0:startIndex] + yList[endIndex:]

            predictions = KNN.predicts(trainingSet_x, trainingSet_y, testingSet_x, k)
            print("\nPredictions: {}".format(predictions))
            print("Testing set Y: {}".format(testingSet_y))
            print("\nAccuracy of fold #{}: {:.2f} %".format(i+1, KNN.evaluate_acc(predictions, testingSet_y)))
           
            avgAcc += KNN.evaluate_acc(predictions, testingSet_y)

            acc.append(KNN.evaluate_acc(predictions, testingSet_y))
            print(acc)

        return avgAcc / ksize
