import numpy as np
import torch
import PrepareDatasetKFold as prepdata
import matplotlib.pyplot as plt
import Performance as per
from sklearn.neighbors import (NearestNeighbors, NeighborhoodComponentsAnalysis, KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold 
from sklearn import metrics

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
<<<<<<< HEAD
    #pca = PCA(n_components=26)
=======
    pca = PCA(n_components=2)
>>>>>>> customLayer
    knn = KNeighborsClassifier(n_neighbors=1) #i 1 or 5
    nca_pipe = Pipeline([('knn',knn)])
    x_train = X[train_i].reshape(X[train_i].shape[0],X[train_i].shape[1]*X[train_i].shape[2]) 
    x_test = X[test_i].reshape(X[test_i].shape[0],X[test_i].shape[1]*X[test_i].shape[2]) 

    y_train, y_test = Y[train_i], Y[test_i]
    
    nca_pipe.fit(x_train, y_train)
    y_pred = nca_pipe.predict(x_test)
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
<<<<<<< HEAD
print("Average Accuracy:", np.mean(Accuracies), '(', np.std(Accuracies),')')
print("Average Recall:", np.mean(Recalls) , '(', np.std(Recalls),')')
print("Average Specificity:", np.mean(Specificities) , '(', np.std(Specificities),')')
print("Average Precision:", np.mean(Precisions) , '(', np.std(Precisions),')')
print("Average Negative Preditive Value:", np.mean(NPVs) , '(', np.std(NPVs),')')
print("Average False Positive Rate:", np.mean(FPRs) , '(', np.std(FPRs),')')
=======
print("Average Accuracy:", np.mean(Accuracies))
print("Average Recall:", np.mean(Recalls))
print("Average Specificity:", np.mean(Specificities))
print("Average Precision:", np.mean(Precisions))
print("Average Negative Preditive Value:", np.mean(NPVs))
print("Average False Positive Rate:", np.mean(FPRs))
>>>>>>> customLayer



 



