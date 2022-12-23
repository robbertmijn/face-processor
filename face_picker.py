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

directory = "../10k US Adult Faces Database/Face Images"
outdir = "../processed_auto"

if not os.path.exists(outdir):
    os.makedirs(outdir)
else:
    pass

dm = io.readtxt("../Full Attribute Scores/demographic & others labels/demographic-others-labels.csv", delimiter = ";")
dm["Image quality"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Image quality"] ]
dm["Age"] = [x.replace(",", ".") if isinstance(x, str) else x for x in dm["Age"] ]
dm["Gender"] = ["m" if x == 1 else "f" for x in dm["Gender"] ]

# get subset based on attributes from csv
subset = (dm["Facial hair?"] == 0) & (dm["Face direction?"] == 1) & (dm["Image quality"] > 2) & (dm["Age"] < 4) & (dm["Age"] > 2) & (dm["Emotion?"] == {0, 1})

i = 0
for g, gdm in ops.split(subset.Gender):
    
    for row in gdm:
        
        path = os.path.join(directory, row.Filename)
        outname = "{}_{}".format(g, i)
        print("Loading {}, ({})".format(path, i))
        img = fp.process_face(path, outdir, outname)
        if img is not None:
            i += 1