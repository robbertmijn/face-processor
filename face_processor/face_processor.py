#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:20:06 2022

@author: robbertmijn


Lines 1-8: Left eyebrow
Lines 9-24: Face shape
Lines 25-32: Right eyebrow
Lines 33-45: Nose
Lines 46-53: Left eye
Lines 54-61: Right eye
Lines 62-70: Upper lip
Lines 71-77: Bottom lip

dlibs facial landmarks are these:
43-48 left eye
37-42 right eye
32-36 nose

face landmarks are included, downloaded from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""

import imutils
import numpy as np
import cv2
import dlib
import os

def process_face(inpath, outdir):
    
    # Get the image from file
    img = cv2.imread(inpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # load the face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join("face_processor", "shape_predictor_68_face_landmarks.dat"))
    
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
    
    # Process the face image
    img = transform_image(img, right_eye, left_eye, nose)
    
    if img is not None:
        outpath = os.path.join(outdir, "processed_{}".format(os.path.basename(inpath)))
        imsaved = cv2.imwrite(outpath, img)
        print(outpath)
        if imsaved:
            print("saving {}".format(outpath))
        else:
            print("saving error")
    else:
        print("can't transform")
        
    return img

def transform_image(img, right_eye, left_eye, nose):
    
    # Hard coded image dimensions
    W = 300
    H = 350
    
    # ignore too large images
    if img.shape[0] > H:
        return None
    
    # Specify mask, conveniently an ellipse the size of the original image
    mask = np.zeros_like(img)
    mask = cv2.ellipse(mask, 
                       (int(img.shape[1]/2), int(img.shape[0]/2)), 
                       (int(img.shape[1]/2) - 3, int(img.shape[0]/2) - 3),
            0, 0, 360, (255, 255, 255), -1)
    
    # Mask the image
    img = np.bitwise_and(img, mask)
    
    # Determine where to shift the image to based on the area between the eyes and nose
    x_shift = int(img.shape[1]/2 - np.array([right_eye[0], left_eye[0], nose[0]]).mean())
    y_shift = int(img.shape[0]/2 - np.array([right_eye[1], left_eye[1], nose[1]]).mean())
    
    # Determine the rotation to get the eyes on the same level horizontally
    rotation = np.degrees(np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])))
    
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
    # Turn mask into logical array (127 as "threshold")
    new_mask = do_transform(mask) < 127
    # new_mask = new_mask < 127
    
    # Reduce contrast by "40"
    contrast = -40
    new_img[~new_mask] = new_img[~new_mask] * (contrast/127+1) - contrast
    new_img = np.clip(new_img, 0, 255)
    
    # On average, each pixel should be 127 (gray: our luminosity aim)
    lum_aim = np.sum([~new_mask]) * 127
    new_img[~new_mask] = new_img[~new_mask] + (lum_aim - new_img[~new_mask].sum()) / np.sum([~new_mask])
    
    # Turn the background gray (127)
    new_img[new_mask] = 127
    
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
