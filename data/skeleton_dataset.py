from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os
import glob
from utils.auxiliaryfunctions import classify, tag_loc, copy_files


class SkeletonDataset(Dataset):
    # subset can be: 'train', 'val', 'test'
    def __init__(self, data_dir, csv_file):
        super(SkeletonDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.files = glob.glob(os.path.join(data_dir, '*.json'))
        self.label_table = pd.read_csv(csv_file)
        self.classes = [0,1,2,3]       
        self.num_sequences = 20

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        poses = []
        cow = os.path.basename(self.files[index]).split('.')[0]
        cow_id = cow.split('-')[0]
        tag = cow.split('-')[1]
        label = classify(self.label_table[(self.label_table['cow_id'] == cow_id)].iloc[0, tag_loc(int(tag))]) - 1

        with open(self.files[index]) as json_file:  
            data = json.load(json_file)
            for frame in data['data']:
                num_frame = frame['frame_index'] - 1
                skeleton = frame['skeleton'][0]
                pose = np.array(skeleton['pose'])
                x_channel = pose[0::2]
                y_channel = pose[1::2]
                if num_frame < self.num_sequences:
                    poses.append(pose)

        poses = np.array(poses)
        sample = {'seq': poses, 'label': label}
             
        return cow, sample
