# -*- coding: utf-8 -*-
"""
Spyder Editor

avc
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path

# This function search for all images in a given directory
def search_images(path, kw):
    """
    Parameters
    ----------
    path : string
        Path to images
    kw : str
        Image type 
       
    Returns
    -------
    list
        list of image filenames


    """
    file_list = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if kw in file:
                file_list.append(os.path.join(r, file))
         
    return(file_list)
    

fnames = search_images('images', '.JPG')
for f in fnames:
    print(f)