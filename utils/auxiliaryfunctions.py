import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import shutil
import os

def tag_loc(tag):
    ''' Return the corresponding location in csv file for the n-th day data'''
    return (tag + 1) if tag != 14 else 9

def classify(score, class_type='lms'):
    ''' Convert locomotion score to label '''
    if class_type == 'lms':
        if score < 2:
            label_index = 1
        elif score >= 2 and score < 3:
            label_index = 2
        elif score >= 3 and score < 4:
            label_index = 3
        elif score >= 4:
            label_index = 4
    elif class_type == 'binary':
        if score <= 2:
            label_index = 1
        elif score > 2:
            label_index = 2
    else:
        print('There is no such classification type.')
    return label_index

def class_distribution(dataset, class_type='lms'):
    num_classes = len(dataset.classes)
    label_table = dataset.label_table
    labels = [[] for i in range(num_classes)]
    for i,  (cow, sample) in enumerate(dataset):
        label = classify(sample['label'], class_type='lms') - 1
        labels[int(label)].append(label)
    print('Class distribution: ', [len(labels[i]) / len(dataset) for i in range(num_classes)])

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
Functions for data augmentation
"""
def scale(poses, mean=1, std=0.15):
    """
    Scale the poses with normal distribution, given mean and std
    """
    poses[:, 0::2] = poses[:, 0::2] * np.random.normal(mean, std)
    
    return poses

def rotate(poses, angle=10):
    """
    Rotate the points counterclockwise by a given angle around a given origin, 
    given the pose sequence: poses (num_seqences, num_features)
    The angle should be given in degrees.
    """
    for f in range(poses.shape[0]):  # Iterate through sequences (frames)
            for i in range(25):      # Iterate through features
                px, py, pz = poses[f, i*3:i*3+3]  # point
                ox, oy, oz = poses[f, 6:9]        # origin (reference point): neck (third in pose)
                # rotation 
                angle = angle * np.pi/180
                qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                qz = pz
                poses[f, i*3:i*3+3] = qx, qy, qz
    
    return poses

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
