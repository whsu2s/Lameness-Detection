import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

video_file = 'data/cow56.mp4'
kernel_dil = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) #np.ones((20, 20), np.uint8)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
cap = cv.VideoCapture(video_file)
fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

frame_width = 720 #1920 #720
frame_height = 385 #1080 #440
# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.mp4', fourcc, 10.0, (frame_width, frame_height), True)

while(1):
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
                print(roi.shape)
 
                # Crop image
                frame_cropped = frame[int(cy - h/2):int(cy + h/2), int(cx - w/2):int(cx + w/2)]
                if (frame_cropped.shape[0] == 0 or frame_cropped.shape[1] == 0):
                    frame_cropped = frame

        """ Remove background """
        '''
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        rect = (x_eff, y_eff, w, h)
        cv.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
        frame = frame * mask2[:,:,np.newaxis]
        '''

        """ Show and save video """
        #img = cv.rectangle(frame, (100,480), (100+600, 480+360), (0,255,255), 5)
        #cv.imshow("Original", frame)
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

