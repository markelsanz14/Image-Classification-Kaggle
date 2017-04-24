from __future__ import division
import numpy as np 
from sklearn.naive_bayes import GaussianNB

def NB(features_train, labels_train, features_test):
    X = features_train
    Y = labels_train
    Z = features_test
    
    clf = GaussianNB()
    clf.fit(X, Y)
    z = clf.predict(Z)
    
    return z

def nb_val(features_train, labels_train, features_val, labels_val):
    W = features_train
    X = labels_train
    Y = features_val
    Z = labels_val
    z = 0
    sumCorrect = 0
    
    clf = GaussianNB()
    clf.fit(W, X)
    b = clf.predict(Y)
    y = b.tolist()
    val_len = len(features_val)
    i = 0
    for value in y:
        if value == Z[i]:
            sumCorrect = sumCorrect+1
        i = i + 1
            
    accuracy = sumCorrect/val_len
    print(sumCorrect)
    result = [b]
    print result
    result.append(accuracy)
    return result
    
