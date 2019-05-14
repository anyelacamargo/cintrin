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


# Crop target object
def crop_object(img, img_thrs):
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
    im2, contours, hierarchy = cv2.findContours(img_thrs, cv2.RETR_TREE, 
                                            cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    x, y, w, h = cv2.boundingRect(img_thrs)
# Getting ROI
    roi = img[y:y + h, x:x + w]
    return(roi)


# Segment object
def filter_frame(img, min, max):
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
    _, thresholded = cv2.threshold(gray, min, max, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25),np.uint8)
    kernel_length = np.array(thresholded).shape[1]//80
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
      
    erosion = cv2.erode(thresholded, kernel, iterations = 1)
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    im_floodfill = closing.copy()
    # Mask used to flood filling.
    h, w = closing.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = closing | im_floodfill_inv
    return(im_floodfill_inv)


#right-click event value 
def click_event(event, x, y, flags, param):
   
    if event == cv2.EVENT_LBUTTONDOWN:
        a = img.shape
        d = list()
        print(a[2])
        for i in range(1, a[2]):
            d.append(img[y,x,i])
       
        print(d)
        


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
      
    return(u)

  
def ccMaxRGB(img):
  [row, col] = img.shape[0:2]
  im2d = img.reshape((row*col, 3))
  im2d = im2double(im2d)
  #MaxRGB
  LMaxRGB = LMaxRGB = list()
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],0]))
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],1]))
  LMaxRGB.append(np.max(im2d[0:im2d.shape[0],2]))
 
  #% illuminant corrected image
  imMaxRGB = im2double(img)
  c=0; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  c=1; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  c=2; imMaxRGB[:,:,c] = imMaxRGB[:,:,c] / LMaxRGB[c];
  
  u = np.uint8(np.round(imMaxRGB*255))
  plt.figure()
  plt.axis("off")
  plt.imshow(u)
  plt.show()
  
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

# Plot image
def plot_image(img):
    plt.figure()
    plt.axis("off")
    plt.imshow(img)
    plt.show()
    
#fnames = search_images('images', '.JPG')
#c = get_features(fnames)
right_clicks = list()   
# Read image
img = cv2.imread('images/C01_ (1).JPG')
# Filter colour card
thres_obj = filter_frame(img, 250, 255)
# Crop colour card
img_cropt = crop_object(img, thres_obj)
#plot_image(img_cropt)

#impixel(img_cropt[:,:,0])
