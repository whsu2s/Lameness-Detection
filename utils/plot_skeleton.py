#!/usr/bin/env python3
# Usage example: python cow_tracking3.py --input data_input/Tag4/cow1404.mp4 --output outputs/crop.mp4

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import argparse
import pandas as pd

from deeplabcut import *
from deeplabcut.utils.video_processor import VideoProcessorCV as vp
from deeplabcut.utils import make_labeled_video
from skimage.draw import circle_perimeter, circle

frame_width = 680
frame_height = 420

##########################
# Take input and output  ################################################## 
##########################
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="take input video for processing") 
parser.add_argument("--output", required=True, help="generate cropped clip")
args = vars(parser.parse_args())
cap = cv2.VideoCapture(args["input"])
input_file = str(args["input"].split('/')[-1].split('.')[0])

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(args["output"], fourcc, 20.0, (frame_width, frame_height), True)
###########################################################################

##########################
# Read hdf files          ################################################## 
##########################
datafile = '/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/label2-fkie-2019-04-30/labeled-data/'
df = pd.read_hdf(os.path.join(datafile, input_file + '/CollectedData_fkie.h5'))
#df = pd.read_hdf(os.path.join(datafile, input_file + '/machinelabels-iter6.h5'))
df = df.iloc[1:]
iframe = np.zeros(df.shape[0])

##########################
# Params for plotting     ################################################## 
##########################
config = '/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/label2-fkie-2019-04-30/config.yaml' 
cfg = auxiliaryfunctions.read_config(config)
bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,'all')
dotsize = cfg['dotsize']
colormap = cfg['colormap']
scorer = 'fkie' 

colorclass=plt.cm.ScalarMappable(cmap=colormap)
C=colorclass.to_rgba(np.linspace(0,1,len(bodyparts)))
colors=(C[:,:3]*255).astype(np.uint8)

# Get frame index
for i in range(df.shape[0]):
    frame = int(df.index[i].split('/')[-1].split('.')[0].split('g')[1])
    iframe[i] = frame
nframes = iframe.shape[0]

df_x = np.empty((len(bodyparts),nframes))
df_y = np.empty((len(bodyparts),nframes))

for bpindex, bp in enumerate(bodyparts):
    df_x[bpindex,:] = df[scorer][bp]['x'].values
    df_y[bpindex,:] = df[scorer][bp]['y'].values

counter = 0
label_counter = 0

while True:
    ret, image_np = cap.read()

    # Plot keypoints
    if counter == iframe[0]:
        iframe = np.delete(iframe, 0)
        
        for bpindex in range(len(bodyparts)):
            xc = int(df_x[bpindex, label_counter])
            yc = int(df_y[bpindex, label_counter]) 
            color = tuple([int(x) for x in colors[len(bodyparts) - bpindex - 1]])
            cv2.circle(image_np, (xc, yc), 3, color, -1)
        label_counter = label_counter + 1
    cv2.imshow('frame', image_np)
    out.write(image_np)

    counter = counter + 1
    
    k = cv2.waitKey(0)
    if k == 32: # press space key to continue
        continue
    elif k == 27:  # press esc key to stop
        break

cap.release()
cv2.destroyAllWindows()
