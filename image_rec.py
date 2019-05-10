# -*- coding: utf-8 -*-
"""
avc
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import os, os.path
from pylab import *

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
    
    
 # This function extract features from images
def get_features(filenames):
    """
    Parameters
    ----------
    filenames : list
        list of image filenames
   
       
    Returns
    -------
    res_imc : dict
        dictionay with image features
    
        


    """
    res_im = dict()

    for f in fnames:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        #plt.imshow(img)
        crop_img = img[500:2000, 1000:3900]
        #plt.imshow(crop_img)
        figure()
        #gray()
    # show contours with origin upper left corner
        contour(crop_img, origin='image')
        axis('equal')
        axis('off')
    
        figure()
        hist(crop_img.flatten(), 128)
        show()
        res_im[f.split("\\")[1]] = np.mean(crop_img.flatten())
             
    return(res_im)
          
        
    

fnames = search_images('images', '.JPG')
c = get_features(fnames)
    
    