#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:20:06 2022

@author: robbertmijn
"""
# import face_processor
from face_processor import face_processor as fp
# import glob
import os
from datamatrix import io, operations as ops
#%% Face selection
"""
From the 2.222 faces we make a subset:
    - exclude facial hair
    - include only eyes front
    - include only happy and neutral
    - include only image quality > 1
    - exclude really old and really young

This should leave 915 faces
"""

# load demographics
dm = io.readtxt("demographic-others-labels.csv", delimiter = ";")
dm["Image quality"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Image quality"] ]
dm["Age"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Age"] ]
dm["Gender"] = ["m" if x == 1 else "f" for x in dm["Gender"] ]

# get subset based on attributes from csv
subset = (dm["Facial hair?"] == 0) & (dm["Face direction?"] == 1) & (dm["Image quality"] > 2) & (dm["Age"] < 4) & (dm["Age"] > 2) & (dm["Emotion?"] == {0, 1})

#%%
orig_dir = "../10k US Adult Faces Database/Face Images"

outdir = "original"
if not os.path.exists(outdir):
    os.makedirs(outdir)
else:
    pass

for g, gdm in ops.split(subset.Gender):
    i = 0
    for row in gdm:
        
        path = os.path.join(orig_dir, row.Filename)
        outname = "{}_{}".format(g, i)
        print("Loading {}, ({})".format(path, i))
        img = fp.process_face(path, outdir, outname)
        if img is not None:
            i += 1




