import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import shutil

def tag_loc(tag):
    ''' Return the corresponding location in csv file for the n-th day data'''
    return (tag + 1) if tag != 14 else 8

def classify(score):
    ''' Convert locomotion score to label '''
    if score < 2:
        label_index = 1
    elif score >= 2 and score < 3:
        label_index = 2
    elif score >= 3 and score < 4:
        label_index = 3
    elif score >= 4:
        label_index = 4
    return label_index

"""
Functions for data preprocessing
"""
def copy_files(src, dst):
    for file in src:
        shutil.copy(file, dst)

""" 
Functions for statistical analysis 
  * cm: confusion matrix created from sklearn.metrics
"""
def accuracy(cm):
    return (cm.trace() / cm.sum())

def precision(cm):
    return (cm.diagonal() / np.sum(cm, axis=0))

def recall(cm):
    return (cm.diagonal() / np.sum(cm, axis=1))

def f_score(cm)

def print_analysis(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print('Confusion matrix:')
    print(cm, type(cm))
    print("Accuracy: ", accuracy(cm))
    print()
    print("label precision recall f-score")
    for label in range(4):
        print("{0:5d} {1:7.3f} {2:8.3f} {3:6.3f}".format(label, precision(cm)[label], recall(cm)[label], 
                                                     f1_score(y_true, y_pred, average=None)[label]))
