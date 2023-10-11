import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import math
import operator
import random
from collections import Counter
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

ionosphere = fetch_ucirepo(id=52) 
X = ionosphere.data.features 
y = ionosphere.data.targets


class KNN:

    def __init__(self, K=3):
        self.K = K          #define k, which is 3 for now
        self.xArray = np.array(X)
        self.yArray = np.array(y)
        return
    
    def fit(self, points):
        self.points = points #we dont need much for fit

    def euclidean(self, p , q):
         return np.sqrt(np.sum(np.array(p) - np.array(q)) ** 2) #euclidean for finding distance between every point in p and q.
    
    def train_test_split(X, y, train_size, shuffle=False):
        length_dataset = len(X)
        length_train = int(length_dataset * train_size)

        # Shuffle dataset x and y in the same way
        if shuffle:
            combine = np.arange(X.shape[0])
            np.random.shuffle(combine)
            X = X[combine]
            y = y[combine]

        # Split as training and test
        X_train = X[:length_train, :]
        X_test = X[length_train:, :]
        y_train = y[:length_train]
        y_test = y[length_train:]

        return X_train, X_test, y_train, y_test
    
    def knn(X_train, y_train, X_test, k):
        temp_prediction = []
        temp_distances = []
        distances = []
        prediction = []
        common = []

        for i in range(len(X_train)):
            for j in range(len(X_test)):
                temp_distances.append([KNN.euclidean(X_train[i], X_test[j]), j])
            temp_distances.sort()
            distances = temp_distances[0:KNN.k]
            for x, y in distances:
                temp_prediction.append(y_train[y])
            prediction.append(Counter(temp_prediction).most_common(1)[0][0])
        return prediction
            
            
        
































































