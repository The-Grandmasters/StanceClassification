import numpy as np
import torch
import PrepareDatasetKFold as prepdata
import matplotlib.pyplot as plt
import Performance as per
from sklearn.model_selection import KFold 
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


X, Y = prepdata.PrepareDataset()
FeatureNames=['fighting','front', 'ready','cat', 'horse', 'hicho','seiza']


Accuracies = []
Precisions = []
Recalls = []
NPVs = []
FPRs = []
Specificities = [] 
kf = KFold(n_splits=10)
for train_i, test_i in kf.split(X):
    x_train = X[train_i].reshape(X[train_i].shape[0],X[train_i].shape[1]*X[train_i].shape[2]) 
    x_test = X[test_i].reshape(X[test_i].shape[0],X[test_i].shape[1]*X[test_i].shape[2]) 
    y_train, y_test = Y[train_i], Y[test_i]


    rf=RandomForestClassifier(n_estimators=45)
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    
    print("-------------------------------")
    ACC, TPR, TNR, PPV, NPV, FPR = per.GetPerformanceMetrics(y_test, y_pred, weighted=True)
    Accuracies.append(ACC)
    Recalls.append(TPR)
    Specificities.append(TNR)
    Precisions.append(PPV)
    NPVs.append(NPV)
    FPRs.append(FPR)

    print("Accuracy: ",ACC)
    print("Recall: ",TPR)
    print("Specificity: ",TNR)
    print("Precision: ",PPV)
    print("Negative Predictive Value: ",NPV)
    print("FP rate(fall-out): ",FPR)
    print( confusion_matrix(y_test, y_pred))

print("-------------------------------")
print("Average Accuracy:", np.mean(Accuracies), '(', np.std(Accuracies),')')
print("Average Recall:", np.mean(Recalls) , '(', np.std(Recalls),')')
print("Average Specificity:", np.mean(Specificities) , '(', np.std(Specificities),')')
print("Average Precision:", np.mean(Precisions) , '(', np.std(Precisions),')')
print("Average Negative Preditive Value:", np.mean(NPVs) , '(', np.std(NPVs),')')
print("Average False Positive Rate:", np.mean(FPRs) , '(', np.std(FPRs),')')




 



