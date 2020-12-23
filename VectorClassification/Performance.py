import numpy as np
from sklearn.metrics import confusion_matrix


#Weighted = true, return weighted acc
def GetPerformanceMetrics(y_test, y_pred, weighted):
    cm = confusion_matrix(y_test, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    weights = cm.sum(axis=0)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    if(weighted):
        TPR = sum(TPR*weights)/sum(weights)
        TNR = sum(TNR*weights)/sum(weights)
        PPV = sum(PPV*weights)/sum(weights)
        NPV = sum(NPV*weights)/sum(weights)
        FPR = sum(FPR*weights)/sum(weights)
        FDR = sum(FDR*weights)/sum(weights)
        ACC = sum(ACC*weights)/sum(weights)

    return ACC, TPR, TNR, PPV, NPV, FPR