# Lameness-Detection
Lameness is a serious disorder in dairy farms that increases the risk of culling of cattle as well as economic losses. This issue is addressed by lameness detection using locomotion scoring, which assesses the lameness level of cattle by their pose and gait patterns. This project aims to evaluate the efficacy of deep neural networks in the context of automated locomotion scoring.

<div align="center">
    <img src="img/workflow.png">
</div>

## Data preparation
The raw data were a series of videos with a resolution of 1920 × 1080 at 50 frames per second (fps). Each video ranges from around four to ten minutes long, containing several cows walking individually on a walkway. These videos were preprocessed to generate the skeleton sequences for lameness detection, as shown in the figure below. 

<div align="center">
    <img src="img/data_overview.png">
</div>

### Video preprocessing
The raw videos were first trimmed into 500 shorter video clips, each of which contains only one cow walking from left to right of the frame. The raw videos were cropped in such a way that the cow is at the center of each frame, which also reduces the size of the data. The processed video clips have a resolution of 680×420 at 20 fps.

### Pose estimation
This step estimates the pose of cows by extracting 25 keypoints (Figure shown below) from their body. An open-source framework called [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) is used for pose estimation. The whole process can be carried out by following the instructions in [DeepLabCut Demo](https://github.com/AlexEMG/DeepLabCut/blob/master/examples/Demo_yourowndata.ipynb). The predicted poses genertaed by DeepLabCut toolbox were transformed into JavaScript Object Notation (JSON) from hierarchical data format (HDF) files to form the skeleton sequence dataset (frames of (x,y) coordinates of keypoints in pixels for each video).

<div align="center">
    <img src="img/keypoints.png">
</div>

## Training

### Approach

