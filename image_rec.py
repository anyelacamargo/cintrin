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
    #im2, contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_TREE, 
     #                                       cv2.CHAIN_APPROX_SIMPLE)
    #sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    x, y, w, h = cv2.boundingRect(img_thrs)
# Getting ROI
    roi = img[y:y + h, x:x + w,:]
    return(roi)


# Segment object
def filter_frame(img_copy, min, max):
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
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, min, max, cv2.THRESH_BINARY)
    
    return(thresholded)
    
#Pre-process whole image   
def postproc_custom1(binay_img):
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
    kernel = np.ones((25, 25),np.uint8)
    erosion = cv2.erode(binay_img, kernel, iterations = 1)
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
        #print(a[2])
        for i in range(1, a[2]):
            d.append(img[y,x,i])
        red = img[y,x,0]
        green = img[y,x,1]
        blue = img[y,x,2]
       
        print(red, green, blue)
        


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
def get_features(f):
    """
    Parameters
    ----------
    f : string
        image filename
   
       
    Returns
    -------
    res_imc : dict
        dictionay with image features
    
        
    """
   
    #img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
    img = cv2.imread(f)
        #plt.imshow(img)
    crop_img = img[500:2000, 1000:3900]
    print(f, np.mean(crop_img.flatten()))
    plot_image(crop_img)
    thres_obj = filter_frame(img, 250, 255)
        # post-process segmented image
    thres_obj = postproc_custom1(thres_obj)
        # Crop colour card
    img_cropt = crop_object(img, thres_obj)
        
        # Filter boxes in color card
    thres_boxes = filter_frame(img_cropt, 230, 255)
    plot_image(thres_boxes)
        # Select boxes
    thres_boxes[np.where(thres_boxes == 0)] = 1
    thres_boxes[np.where(thres_boxes == 255)] = 0
    kernel = np.ones((25, 25),np.uint8)
    thres_boxes = cv2.erode(thres_boxes, kernel, iterations = 1)
    
    im_floodfill = thres_boxes.copy()
    # Mask used to flood filling.
    h, w = thres_boxes.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    # Invert floodfilled image
    thres_boxes = cv2.bitwise_not(im_floodfill)
       
    #plot_image(thres_boxes)
        
    im2, contours, hierarchy = cv2.findContours(thres_boxes, cv2.RETR_TREE, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        
    box_dic = dict()
    for i in range(0, len(contours)) :
        roi = crop_object(img_cropt, contours[i])
        #plot_image(roi)
        box_dic[i] = np.mean(roi.flatten()), cv2.contourArea(contours[i])
            
    #plot_image(img_cropt)
       
             
    return(np.mean(crop_img.flatten()), box_dic)
          

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
    """
    Parameters
    ----------
    img : image (RGB)
        image
           
    """
    plt.figure()
    plt.axis("on")
    plt.imshow(img)
    plt.show()



fnames = search_images('images', '.JPG')
res_im = dict()

for f in fnames:
    res_im[f.split("\\")[1]] = get_features(f)
     
print (res_im)



fout = "imag_res.csv"    
fo = open(fout, "w")
s = "filename" +  "," + "col" + "," + "row" + "," + "plotcolor" 
for i in range(0, 6):
    s = s + ", " + 'box' + str(i)
s = s + "\n"
fo.write(s)
    
    
for k in res_im.keys():
    s = ""
    s = s + k + "," + str(k.split("_")[0]) + \
    "," + str(k.split("_")[1].split(")")[0].split("(")[1]) + "," + \
    str(res_im[k][0])
    for k1 in range(0, len(res_im[k][1])):
        s = s + ","  +  str(res_im[k][1][k1][0])
    s = s + "\n"
    fo.write(s)
    
fo.close()


