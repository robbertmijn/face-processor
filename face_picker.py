#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:20:06 2022

@author: robbertmijn
"""

from face_processor import face_processor as fp
import os
from datamatrix import io, operations as ops
import glob

#%% Face selection

"""
From the 2.222 faces we make a subset:
    - exclude facial hair
    - include only eyes front
    - include only happy and neutral
    - include only image quality > 1
    - exclude really old and really young

This should leave 915 faces

Filenames and attributes of these faces are stored in a csv file. The actual files are stored in a separate directory.
"""

# load demographics
dm = io.readtxt("demographic-others-labels.csv", delimiter = ";")
dm["Image quality"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Image quality"] ]
dm["Age"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Age"] ]
dm["Gender"] = ["m" if x == 1 else "f" for x in dm["Gender"] ]

# get subset based on attributes from csv
subset = (dm["Facial hair?"] == 0) & (dm["Face direction?"] == 1) & (dm["Image quality"] > 2) & (dm["Age"] < 4) & (dm["Age"] > 2) & (dm["Emotion?"] == {0, 1})

#%% Process the faces from the subset we just created and store them

faces_dir = "../10k US Adult Faces Database/Face Images" # This is where I have them stored
outdir_faces = "faces"

# create directories if needed
if not os.path.exists(outdir_faces):
    os.makedirs(outdir_faces)
else:
    pass

# go over the list of subset images and create new images (split for gender, which we need for our experiment)
for g, gdm in ops.split(subset.Gender):
    i = 0
    for row in gdm:      
        path = os.path.join(faces_dir, row.Filename)
        outname = "{}_{}".format(g, i)
        print("Loading {}, ({})".format(path, i))
        img = fp.process_face(path, outdir_faces, outname)
        if img is not None:
            i += 1
            
#%% Process faces that are stored separately in a directory (e.g., probes)

probes_dir = "probes_raw"
outdir_probes = "probes"

if not os.path.exists(outdir_probes):
    os.makedirs(outdir_probes)
else:
    pass

i = 0
for path in glob.glob(os.path.join(probes_dir, "*.jpg")):
    outname = "probe_{}".format(i)
    print("Loading {}, ({})".format(path, i))
    img = fp.process_face(path, outdir_probes, outname)
    if img is not None:
        i += 1
        
#%% Process images without faces

img = fp.process_face("probe_1_raw.jpg", ".", "probe_1.jpg")



