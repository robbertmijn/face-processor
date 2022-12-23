#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:20:06 2022

@author: robbertmijn
"""
from face_processor import face_processor as fp
import glob
import os

directory = "test_images"
outdir = "processed_auto"

if not os.path.exists(outdir):
    os.makedirs(outdir)
else:
    pass

paths = glob.glob(os.path.join(directory, "*.jpg"))

for i, path in enumerate(paths[0:100]):
    
    # print("Loading {}, ({})".format(path, i))
    fp.process_face(path, outdir)