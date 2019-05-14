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



import argparse
import utils
from collections import Counter
import imutils
import pprint
import gc
from skimage import measure
import urllib
from win32api import GetSystemMetrics

# Segment object
def filter_frame(img):
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    kernel = np.ones((20, 20),np.uint8)
    kernel_length = np.array(thresholded).shape[1]//80
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
      
    erosion = cv2.erode(thresholded, kernel, iterations = 1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    #dilation = cv2.dilate(closing, kernel,iterations = 6)
    #dilation = cv2.dilate(dilation, verticle_kernel,iterations = 4)
    #dilation = cv2.erode(dilation, verticle_kernel,iterations = 4)
     
    plt.figure()
    plt.axis("off")
    plt.imshow(closing)
    plt.show()
      
    return(thresholded)

#right-click event value 
def click_event(event, x, y, flags, param):
    global right_clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        red = img[y,x,2]
        blue = img[y,x,0]
        green = img[y,x,1]
        
        #store the coordinates of the right-click event
        right_clicks.append([x, y])
        #print(red, green, blue) 
        print(right_clicks)
        


# Convert image to double
def im2double(img):
    """
     Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D

    """  
    info = np.iinfo(img.dtype) # Get the data type of the input image
    out = img.astype(np.float) / info.max
    return out


def ccGrayWorld(img):
  
  [row, col] = img.shape[0:2]
  im2d = img.reshape((row*col, 3))
  im2d = im2double(im2d)
  # #Grey World
  # illuminant corrected image
  imGW = im2double(img)
  c=0; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
  c=1; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
  c=2; imGW[:,:,c] = imGW[:,:,c] / cv2.mean(im2d[0:im2d.shape[0], c])[0]
  u = np.uint8(np.round(imGW*255))
  #plt.figure()
  #plt.axis("off")
  #plt.imshow(u)
  #plt.show()
  return(u)
  


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
          

# Get pixel value      
def impixel(img):
    """
     Parameters
    ----------
    img : image (RGB)
        image
           
    Returns
    -------
    out
        image 2D
    """
    
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    #
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.setMouseCallback("image", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows


#fnames = search_images('images', '.JPG')
#c = get_features(fnames)
    
right_clicks = list()
img = cv2.imread('images/C01_ (1).JPG')
#coco = ccGrayWorld(img)

#gray = cv2.cvtColor(coco, cv2.COLOR_BGR2GRAY)
filter_frame(img)
