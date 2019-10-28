#!/usr/bin/env python3
# Usage example: python3 create_skeleton.py --input ~/Documents/Thesis/src/annotation/data/all2/ --output ~/Documents/Thesis/src/annotation/data/all2/data_json/

import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

from deeplabcut import *
from deeplabcut.utils.video_processor import VideoProcessorCV as vp

import sys  
sys.path.append('/home/wei-chan.hsu/Dokumente/Thesis/utils')  
from auxiliaryfunctions import classify, tag_loc

frame_width = 680
frame_height = 420
fps = 20
##########################
# Take input and output  ################################################## 
##########################
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="take input video for processing") 
parser.add_argument("--output", required=True, help="generate cropped clip")
args = vars(parser.parse_args())
input_path = str(args["input"])
output_path = str(args["output"])

##########################
# Read hdf files          ################################################## 
##########################
datafile = '/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/label2-fkie-2019-04-30/labeled-data/'
#df = pd.read_hdf(os.path.join(datafile, input_file + '/machinelabels-iter6.h5'))

##########################
# Read video files        ################################################## 
##########################
videos = input_path 
Videos = auxiliaryfunctions.GetVideoList('all',videos,'mp4')

##########################
# Params for plotting     ################################################## 
##########################
config = '/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/label2-fkie-2019-04-30/config.yaml' 
cfg = auxiliaryfunctions.read_config(config)
bodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(cfg,'all')
scorer = 'fkie' 

for video in Videos:
    vname = str(Path(video).stem)
    #print(vname)
    Dataframe = pd.read_hdf(os.path.join(datafile, vname + '/CollectedData_fkie.h5'))
    df = Dataframe.iloc[1:]    # Remove first frame
    iframe = np.zeros(df.shape[0])
    data = []
    counter = 0
    label_counter = 0

    # Get frame index
    for i in range(df.shape[0]):
        frame = int(df.index[i].split('/')[-1].split('.')[0].split('g')[1])
        iframe[i] = frame
    nframes = iframe.shape[0]

    # Extract labels from file
    label_table = pd.read_csv('/home/wei-chan.hsu/Dokumente/Thesis/src/annotation/data_labels.csv')
    cow_id = vname.split('-')[0]
    tag = vname.split('-')[1]
    label = label_table[(label_table['cow_id'] == cow_id)].iloc[0, tag_loc(int(tag))]
    df_x = np.empty((len(bodyparts),nframes))
    df_y = np.empty((len(bodyparts),nframes))

    for bpindex, bp in enumerate(bodyparts):
        df_x[bpindex,:] = df[scorer][bp]['x'].values
        df_y[bpindex,:] = df[scorer][bp]['y'].values

    # Get keypoints from frames
    while len(iframe) != 0:
        if counter == iframe[0]:
            iframe = np.delete(iframe, 0)
            # Set neck as reference point
            xr = int(df_x[2, label_counter])
            yr = int(df_y[2, label_counter])
            poses = []
        
            for bpindex in range(len(bodyparts)):
                xc = int(df_x[bpindex, label_counter])
                yc = int(df_y[bpindex, label_counter]) 
                # Concatenate coordinates of keypoints
                poses.extend([round((xc-xr) / frame_width, 3), round((yc-yr) / frame_height, 3)])

            # Create data format
            data.append({"frame_index": counter, "skeleton": [{"pose": poses}]})
            label_counter = label_counter + 1
        else:
            counter = counter + 1

    data = {"data":data, "label": label, "label_index": classify(label)}
    with open(output_path + "{}.json".format(vname), "w") as write_file:
        json.dump(data, write_file)
    
