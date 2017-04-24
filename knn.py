from __future__ import division
import numpy as np 
from sklearn import neighbors
from sklearn.metrics import accuracy_score

def knn_val(features_train, labels_train, features_val, labels_val):
    x = features_train
    y = labels_train
    x_val = features_val
    y_val = labels_val
    z = 0
    sumCorrect = 0
    
    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(x, y)
    y_predict = knn.predict(x_val)
    #y_predict = b.tolist()
    #val_len = len(features_val)
    #print(len(y_predict))
    #print(len(labels_val))
    #print(y_predict)
    #print(labels_val)
    #return
    #tot = 0
    #for i in range(len(y_predict)):
    #    if y_predict[i] == labels_val[i]:
    #        tot += 1

    return accuracy_score(labels_val, y_predict)#float(tot)/len(y_predict)
    '''
    i = 0
    for array in y:
        try:
            temp = array.index(1)
            pass
        except ValueError:
            temp = 9
        if temp == Z[i]:
            sumCorrect = sumCorrect+1
        i = i + 1
            
    accuracy = sumCorrect/val_len
    print(sumCorrect)
    result = [b]
    print result
    result.append(accuracy)
    return result
    '''
