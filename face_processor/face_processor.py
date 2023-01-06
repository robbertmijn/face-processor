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
from matplotlib import pyplot as plt

# load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join("face_processor", "shape_predictor_68_face_landmarks.dat"))

# mean H/W ratio is 1.27
W = 400
H = int(W * 1.27)
CROP = .75
    
def process_face(inpath, outdir, outname, debug = False):
    
    # Get the image from file and force to size
    img = cv2.imread(inpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
    new_img = _img_to_size(img, W, H)
    
    # detect faces
    dets = detector(new_img, 1)   
    if len(dets) != 1:
        # if 0 or more than 1 faces are detected we stop
        return None
    
    # extract landmarks from image
    landmarks = _shape_to_np(predictor(new_img, dets[0]))    
    left_eye, right_eye, nose = _get_tri_points(landmarks)
    
    # Transform the image (shift, rotate, mask)
    new_img = _transform_image(new_img, right_eye, left_eye, nose)

    # Save new image
    if new_img is not None:
        
        outpath = os.path.join(outdir, "{}.jpg".format(outname))
        imsaved = cv2.imwrite(outpath, new_img)

        if imsaved:
            print("saving {}".format(outpath))
        else:
            print("saving error")
            
    else:
        print("can't transform")       

    return new_img
    
    
def _transform_image(im, right_eye, left_eye, nose):
    
    # Determine where to shift the image to based on the area between the eyes and nose
    x_shift = int(im.shape[1]/2 - np.array([right_eye[0], left_eye[0], nose[0]]).mean())
    y_shift = int(im.shape[0]/2 - np.array([right_eye[1], left_eye[1], nose[1]]).mean())
    
    # Determine the rotation to get the eyes on the same level horizontally
    rotation = np.degrees(np.arctan((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0])))
    
    # TODO scale factor based on facial features
    # x, y = zip(left_eye, right_eye, nose)
    # scale = FACE_AREA / get_area(x, y)
    
    im = _do_transform(im, x_shift, y_shift, rotation)
    
    im, mask = _mask_image(im)
    
    im = _set_contrast_and_luminance(im, mask)       

    # Turn the background gray (127)
    im[mask < 127] = 127
    
    # if we had to shift on the horizontal axis too much, we discard
    if abs(x_shift) > 32:
        print("off center by {}".format(x_shift))
        return None
    
    return im


def _shape_to_np(shape, dtype = "int"):
    
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
    
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
        
	# return the list of (x, y)-coordinates
	return coords    


def _get_area(x, y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def _get_tri_points(landmarks):
    
    # Extract center-points of both eyes and the nose
    left_eye = [landmarks[42:47, 0].mean(), landmarks[42:47, 1].mean()]
    right_eye = [landmarks[36:41, 0].mean(), landmarks[36:41, 1].mean()]
    nose = [landmarks[27:35, 0].mean(), landmarks[27:35, 1].mean()]
    
    return left_eye, right_eye, nose
    
def _img_to_size(im, W, H):
    
    # scale to height
    scale = H / im.shape[0]  
    
    dims = (int(im.shape[1] * scale), H)
    
    im = cv2.resize(im, dims, interpolation = cv2.INTER_AREA)
    im = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value = 0)
    
    strt_idx_W = (im.shape[1] // 2) - (W // 2)
    end_idx_W = (im.shape[1] // 2) + (W // 2)

    strt_idx_H = (im.shape[0] // 2) - (H // 2)
    end_idx_H = (im.shape[0] // 2) + (H // 2)
    
    im = im[strt_idx_H:end_idx_H, strt_idx_W:end_idx_W]
    
    return im


def _mask_image(im):
    
    # Mask the image
    mask = np.zeros_like(im)
    mask = cv2.ellipse(mask, 
                        (int(W/2), int(H/2)), 
                        (int(W/2 * CROP), int(H/2 * CROP)),
            0, 0, 360, (255, 255, 255), -1)
    im = np.bitwise_and(im, mask)
     
    return im, mask
   

def _set_contrast_and_luminance(im, mask):
    
    # Reduce contrast by "50"
    # TODO use equalizeHist or CLAHE to equalize contrast
    contrast = -50
    im[mask > 127] = im[mask > 127] * (contrast/127+1) - contrast
    im[mask > 127] = np.clip(im[mask > 127], 0, 255)
    
    # On average, each pixel should be 127 (gray: our luminosity aim)
    lum_aim = np.sum(mask) * 127
    im[mask > 127] = im[mask > 127] + (lum_aim - im[mask > 127].sum()) / np.sum([mask > 127])

    return im

def _do_transform(im, x_shift, y_shift, rotation):
    
    # Pad and add extra room to accomodate the rotation
    im = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value = 0)
    
    im = imutils.rotate(im, angle = rotation)
    im = imutils.translate(im, x_shift, y_shift)
    
    im = im[100:-100, 100:-100]
    
    return im
    
    
def plt_landmarks(img, landmarks, rect):
   
   left_eye, right_eye, nose = _get_tri_points(landmarks) 
   
    # add landmarks
   for lm in landmarks:        
        img = cv2.circle(img, lm, 2, 255, -1)

   img = cv2.rectangle(img, (rect.left(), rect.top()), 
                       (rect.right(), rect.bottom()), 255)
    
   pts = np.array([left_eye, right_eye, nose], np.int32)
   img = cv2.polylines(img, [pts], True, 255, 2)
    
   plt.imshow(img)