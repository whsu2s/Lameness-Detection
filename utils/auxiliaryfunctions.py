import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import shutil
import os

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

def categorize_data(dataset, label_table):   
    for file in dataset:
        #files.append(os.path.splitext(f)[0])
        path = os.path.dirname(file)
        filename = os.path.basename(file).split('.')[0]
        cow_id = filename.split('-')[0]
        tag = (filename.split('-')[1]).split('s')[0]
        label = classify(label_table[(label_table['cow_id'] == cow_id)].iloc[0, tag_loc(int(tag))])

        if label == 1:
            shutil.move(file, path + '/LS1')
        elif label == 2:
            shutil.move(file, path + '/LS2')
        elif label == 3:
            shutil.move(file, path + '/LS3')
        elif label == 4:
            shutil.move(file, path + '/LS4')

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

def f_score(cm):
    p = precision(cm)
    r = recall(cm)
    return 2 * (p * r) / (p + r)

def print_analysis(cm):
    #cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print('Confusion matrix:')
    print(cm)
    print("Accuracy: ", accuracy(cm))
    print()
    print("label precision recall f-score")
    for label in range(cm.shape[0]):
        print("{0:5d} {1:7.3f} {2:8.3f} {3:6.3f}".format(label, precision(cm)[label], recall(cm)[label], 
                                                     f_score(cm)[label]))
