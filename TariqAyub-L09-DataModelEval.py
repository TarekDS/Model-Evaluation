# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:32:27 2019

@author: T
"""

import numpy as np
from sklearn.metrics import *

# A = actual labels, y = probability output of classifier
A = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1])
y = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
Y = np.round(y, 0)


# Confusion Matrix
CM = confusion_matrix(A, Y)
print ("\n\nConfusion matrix:\n", CM)
#True positive, True Negative, False positive, False Negative
tn, fp, fn, tp = CM.ravel()
print ("\nTP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
# Accuracy rate
AR = accuracy_score(A, Y)
print ("\nAccuracy rate:", AR)
#error rate
ER = 1.0 - AR
print ("\nError rate:", ER)
#precision Score
P = precision_score(A, Y)
print ("\nPrecision:", np.round(P, 2))
#recall
R = recall_score(A, Y)
print ("\nRecall:", np.round(R, 2))
#F1 score
F1 = f1_score(A, Y)
print ("\nF1 score:", np.round(F1, 2))

# ROC analysis
 # False Positive Rate, True Posisive Rate, probability thresholds
fpr, tpr, th = roc_curve(A, y)
AUC = auc(fpr, tpr)
print ("\nTP rates:", np.round(tpr, 2))
print ("\nFP rates:", np.round(fpr, 2))
print ("\nProbability thresholds:", np.round(th, 2))
#AUC Score
print ("\nAUC score (using auc function):", np.round(AUC, 2))
print ("\nAUC score (using roc_auc_score function):", np.round(roc_auc_score(A, y), 2), "\n")
