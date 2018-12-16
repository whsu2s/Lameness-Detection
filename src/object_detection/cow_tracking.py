# Resources:
# Object detection: https://github.com/tensorflow/models/tree/master/research/object_detection
# Kalman filter: https://github.com/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb?create=1 

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image

import cv2
video_file = 'test_data/cow1.mp4'
cap = cv2.VideoCapture(video_file)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation
# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here.
# See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# Simple model: 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28' 

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

centroids = []
c0 = (108, 680)
frame_width = 1920
frame_height = 1080
box_width = 680
box_height = 420
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('crop.mp4', fourcc, 20.0, (box_width, box_height), True)

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `21`, we know that this corresponds to `cow`.
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Calculate centroid using exponentail moving average
def calculate_centroid(coordinates, n):
    # Get coordinates from modified visualize_boxes_and_labels_on_image_array()
    yt, xl, yb, xr = coordinates
    cx_pre, cy_pre = np.array(centroids)[-1] if len(centroids) > 0 else c0
    cx = frame_width * (xl + xr) / 2 if (xl != 0 and xr != 0) else cx_pre
    cy = c0[1] #frame_height * (yt + yb) / 2 if (yt != 0 and yb != 0) else cy_pre
    centroids.append((cx, cy))
    if len(centroids) > n:
        centroids.pop(0)
    nSamples = len(centroids)
    #cx, cy = cx_pre, cy_pre
    # Exponential moving average
    ema_x, ema_y = cx_pre, cy_pre
    K = 2 / (nSamples + 1)
    multiplier = 2/(nSamples + 1)
    if nSamples == 1:
        ema_x, ema_y = cx, cy
    else:
        ema_x = (np.array(centroids)[-1, 0] - ema_x) * multiplier + ema_x
        ema_y = (np.array(centroids)[-1, 1] - ema_y) * multiplier + ema_y
    cx, cy = ema_x, ema_y

    return cx, cy

# Tracking using Kalman filter
class KalmanFilter(object):
    def __init__(self, c0):
        ''' Define parameters '''
        dt = 0.02
        self.xp = np.hstack((c0, np.array([10.0, 0.0]))).T       # Initial state
        self.xp = self.xp.reshape(4,1)
        self.x = np.zeros((2,1))
        self.A = np.array([[1.0, 0.0, dt, 0.0],                  # Dynamic matrix
                           [0.0, 1.0, 0.0, dt],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])
        self.P = 10*np.diag([10.0, 10.0, 10.0, 10.0])            # Uncertainty matrix
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])                # Measurement matrix
        self.r = 10.0**2
        self.R = np.array([[self.r, 0.0], [0.0, self.r]])            # Measurement noise covariance
        self.sa = 100
        self.G = np.array([[0.5*dt**2], [0.5*dt**2], [dt], [dt]])
        self.Q = np.dot(self.G, self.G.T) * self.sa **2              # Process noise covariance

    def predict(self):
        self.x = np.dot(self.A, self.xp)                             # Prediction
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        self.xp = self.x
        return self.x

    def update(self, measurement):
        I = np.eye(4)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R         # Update measurement
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.pinv(S)))

        self.Z = measurement.reshape(2,1)                             # Update prediction
        self.y = self.Z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, self.y)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)              # Update error covariance
        self.xp = self.x
        #self.x[1] = measurement[1]
        return self.x


# # Detection

# Initialize filter
kf = KalmanFilter(c0)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Visualization of the results of a detection.
            im, coordinates = vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                skip_boxes=True)

            # Estimate centroid
            cx, cy = calculate_centroid(coordinates, 2)

            # Kalman Filtering
            measurement = np.array([cx, cy])
            prediction = kf.predict()
            prediction = kf.update(measurement)
            cx, cy = prediction[:2]

            cv2.circle(image_np, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.rectangle(image_np, (int(cx - box_width/2), int(cy - box_height/2)), (int(cx + box_width/2), int(cy + box_height/2)), (0,255,0), 2)

            # ROI
            if (cx > box_width / 2) and (cx < frame_width - box_width / 2):
                cropped_img = image_np[int(cy - box_height/2):int(cy + box_height/2), int(cx - box_width/2):int(cx + box_width/2)]
                cv2.imshow('cropped', cropped_img)
                out.write(cropped_img)

            cv2.imshow('object detection', image_np)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

cap.release()
cv2.destroyAllWindows()
