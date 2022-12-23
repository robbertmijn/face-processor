#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:20:06 2022

@author: robbertmijn

dlibs facial landmarks are these:
43-48 left eye
37-42 right eye
32-36 nose

dlibs face landmarks are included, downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import imutils
import numpy as np
import cv2
import dlib
import os

# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join("face_processor", "shape_predictor_68_face_landmarks.dat"))

def process_face(inpath, outdir):
    
    # Get the image from file
    img = cv2.imread(inpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect faces
    dets = detector(img, 1)
    
    if len(dets) != 1:
        # if 0 or more than 1 faces are detected we stop
        return None
    
    # extract landmarks from image
    landmarks = shape_to_np(predictor(img, dets[0]))
    
    # Extract center-points of both eyes and the nose
    left_eye = [landmarks[42:47, 0].mean(), landmarks[42:47, 1].mean()]
    right_eye = [landmarks[36:41, 0].mean(), landmarks[36:41, 1].mean()]
    nose = [landmarks[27:35, 0].mean(), landmarks[27:35, 1].mean()]
    face_rect = dets[0]
    
    # # add landmarks
    # for lm in landmarks:        
    #     img = cv2.circle(img, lm, 2, 255, -1)
    #     img = cv2.rectangle(img, (dets[0].left(), dets[0].top()), 
    #                         (dets[0].right(), dets[0].bottom()), 255)
    
    # Process the face image
    img = transform_image(img, right_eye, left_eye, nose, face_rect)
    
    if img is not None:
        outpath = os.path.join(outdir, "{}".format(os.path.basename(inpath)))
        imsaved = cv2.imwrite(outpath, img)
        print(outpath)
        if imsaved:
            pass
            # print("saving {}".format(outpath))
        else:
            print("saving error")
    else:
        print("can't transform")
        
    return img

def transform_image(img, right_eye, left_eye, nose, face_rect):
    
    # Hard coded image dimensions
    W = 300
    H = 350
    
    # ignore too large images
    if img.shape[0] > H:
        return None
        
    # Determine where to shift the image to based on the area between the eyes and nose
    x_shift = int(img.shape[1]/2 - np.array([right_eye[0], left_eye[0], nose[0]]).mean())
    y_shift = int(img.shape[0]/2 - np.array([right_eye[1], left_eye[1], nose[1]]).mean())
    
    # Determine the rotation to get the eyes on the same level horizontally
    rotation = np.degrees(np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])))
    
    # TODO size transformation
    
    # Determine how much padding is required to get the desired size of the image
    pad_x = int((W - img.shape[1])/2)
    pad_y = int((H - img.shape[0])/2)
    
    # Specify the transformation (The mask must be transformed too!)
    def do_transform(im):
        # Pad and add extra room to accomodate the rotation
        im = cv2.copyMakeBorder(im, 100 + pad_y, 100 + pad_y, 100 + pad_x, 100 + pad_x, cv2.BORDER_CONSTANT, value = 0)
        im = imutils.rotate(im, angle = rotation)
        im = imutils.translate(im, x_shift, y_shift)
        # Remove the extra room
        im = im[100:(H+100), 100:(W+100)]
        return im
    
    new_img = do_transform(img)
    
    # Mask the image
    # TODO mask is now hard coded to be 90 by 70, make this variable
    mask = np.zeros_like(new_img)
    mask = cv2.ellipse(mask, 
                        (int(W/2), int(H/2)), 
                        (70, 100),
            0, 0, 360, (255, 255, 255), -1)
    new_img = np.bitwise_and(new_img, mask)
     
    # Reduce contrast by "50"
    # TODO use equalizeHist or CLAHE to equalize contrast
    contrast = -50
    new_img[mask > 127] = new_img[mask > 127] * (contrast/127+1) - contrast
    new_img[mask > 127] = np.clip(new_img[mask > 127], 0, 255)
    
    # On average, each pixel should be 127 (gray: our luminosity aim)
    lum_aim = np.sum(mask) * 127
    new_img[mask > 127] = new_img[mask > 127] + (lum_aim - new_img[mask > 127].sum()) / np.sum([mask > 127])
    
    # # Output mean luminance and standard deviation (metric for how much contrast there is)
    # print("after, std {}, mean {}".format(new_img[mask > 127].std(), new_img[mask > 127].mean()))
    
    # Turn the background gray (127)
    new_img[mask < 127] = 127
    
    # if we had to shift on the horizontal axis too much, we discard
    if abs(x_shift) > 32:
        print("off center by {}".format(x_shift))
        return None
    
    return new_img

def shape_to_np(shape, dtype="int"):
    
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
    
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
        
	# return the list of (x, y)-coordinates
	return coords    
