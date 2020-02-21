# Lameness-Detection
Lameness is a serious disorder in dairy farms that increases the risk of culling of cattle as well as economic losses. This issue is addressed by lameness detection using locomotion scoring, which assesses the lameness level of cattle by their pose and gait patterns. This project aims to evaluate the efficacy of deep neural networks in the context of automated locomotion scoring, with a comparison with other machine learning methods.

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

## Approach
A hierarchical recurrent neural network ([HRNN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.html)) was adopted to train the skeleton data for lameness detection. Two other machine learning methods were used for comparison.

### Hierarchical Recurrent Neural Network ([HRNN](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.html))
The network is based on the idea that actions are dependent on the movements of individual body parts and their combinations. The coordinates of the five body parts are fed into five subnets, and the representations of each part are hierarchically fused as the number of layers increases.

<p align="center">
    <img align="center" src="img/hrnn.png" | width=700>
</p>

### Random Foreset
To apply the random forest, several features were selected: right/left step overlaps, strides, back arch. These features were calculated from the skeleton data across the time sequence for each data sample (cow). 

<p align="center">
    <img src="img/features.png" | width=300>
</p>

### K-Clustering (Unsupervised Learning)
The reason of applying unsupervised learning is to avoid the issue of incorrect manual labeling (locomotion scores).

## Result

## Discussion
After the analysis of results, the issues of this project are sumarized below:
1. Data: Both the data amount and quality play a significant role.  

2. Method: The adoption of HRNN is based on the assumption that the features of each body part and their correlation are important to detect lameness. The method can be improved by including another stream of cow's whole body, such that the model can learn features not only from individual body parts but also from the whole appearance.

3. Pose estimation
