# Import modules
import numpy as np
import cv2 as cv
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

#  Imports from the object detection module.
from utils import label_map_util
from utils import visualization_utils as vis_util

# Define the codec and create VideoWriter object
frame_width = 1920 #720
frame_height = 1080 #440
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height), True)

''' Model '''
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map2.pbtxt')


''' Load a (frozen) Tensorflow model into memory. '''
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

''' Helper functions '''
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


''' Detection '''
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
#PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', \
            'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

''' Detection '''

video_file = 'test_data/cow1.mp4'
cap = cv.VideoCapture(video_file)

while(True):
    ret, frame = cap.read()
    fshape = frame.shape   # (1080, 1920, 3)
    #frame = frame[100:fshape[0] - 100, :fshape[1] - 100, :]

    if (ret == True):

        """ Show and save video """
        #img = cv.rectangle(frame, (100,480), (100+600, 480+360), (0,255,255), 5)
        #cv.imshow("Original", frame)

        frame_np = load_image_into_numpy_array(Image.fromarray(frame))
        frame_np_expanded = np.expand_dims(frame_np, axis=0)
        output_dict = run_inference_for_single_image(frame_np, detection_graph)
    # Visualization of the results of a detection.

        vis_util.visualize_boxes_and_labels_on_image_array(
          frame_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(frame_np)
        #cv.imshow("Result", frame_np)
        #out.write(frame_np)

    #k = cv.waitKey(0) & 0xff
    #if k == 27:
    #    break
out.release()
cap.release()
cv.destroyAllWindows()

'''
for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)

'''


'''
video_file = 'test_data/cow1.mp4'
kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #np.ones((20, 20), np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cap = cv.VideoCapture(video_file)
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

frame_width = 720 #1920 #720
frame_height = 385 #1080 #440
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height), True)

while(True):
    ret, frame = cap.read()
    fshape = frame.shape   # (1080, 1920, 3)
    #frame = frame[100:fshape[0] - 100, :fshape[1] - 100, :]

    if (ret == True):
        """ Initial background subtraction for detection """
        fgmask = fgbg.apply(frame)
        #cv.imshow("Foreground", fgmask)

        # Filtering
        fgmask = cv.medianBlur(fgmask, 5)

        """ Object detection by contours """
        # Dilation and erosion
        fgmask = cv.morphologyEx(fgmask, cv.MORPH_CLOSE, kernel)
        #cv.imshow("Open", fgmask)
        dilation = cv.dilate(fgmask, kernel_dil, iterations = 3)
        #cv.imshow("Dilation", dilation)

        # Contours
        _, contours, hierarchy = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv.contourArea(contour)
            if (area > 30000):          # Consider only large area contour
                # Moments and centroid
                M = cv.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    #print('centroid: ', cx,cy)
                else:
                    cx, cy = 0, 0
                # Draw bounding box
                x,y,w,h = cv.boundingRect(contour)
                w, h = 720, 440
                y = 500
                #x = (int(cx - w/2) if cx > w/2 else x)
                #img = cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                x_eff, y_eff = int(cx - w/2), int(cy - h/2)
                img = cv.rectangle(frame, (x_eff, y_eff), (int(cx + w/2), int(cy + h/2)), (0,255,0), 2)
                roi = frame[y:y-60+h+5, x:x-10+w+10]
                #print(roi.shape)

                # Crop image
                frame_cropped = frame[int(cy - h/2):int(cy + h/2), int(cx - w/2):int(cx + w/2)]
                if (frame_cropped.shape[0] == 0 or frame_cropped.shape[1] == 0):
                    frame_cropped = frame

        """ Remove background """

        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (x_eff, y_eff, w, h)
        cv.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
        frame = frame * mask2[:,:,np.newaxis]


        """ Show and save video """
        #img = cv.rectangle(frame, (100,480), (100+600, 480+360), (0,255,255), 5)
        cv.imshow("Original", frame)
        # Show cropped image / region of interest
        cv.imshow("Cropped", roi)

        # Save frames to file
        out.write(roi)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
out.release()
cap.release()
cv.destroyAllWindows()
'''
