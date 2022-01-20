#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:57:17 2018

@author: timothylucas
"""

import numpy as np
import os
from PIL import Image
import math

## Preprocessing and visualising the dataset

def convertImageToU8bit(input_img):
    # assumes input array that can be represented 
    # as an image, needs to be scaled to a [0, 255]
    # range, which is what this function returns
    input_img -= np.min(input_img)
    input_img *= (255.0/input_img.max())
    return input_img

def loadFaces(path):
    files = [f for f in os.listdir(path) if ( ('jpeg' in f) | ('jpg' in f) | ('png' in f) )]
    N = len(files)
    
    # Use a dict to store everything in the end
    all_faces = []
    S = []

    for i in range(N):
        im = Image.open(path+files[i])
        # convert to grayscale
        im = im.convert('L')
        # make sure it's the right size
        if im.size != (300, 300):
            size = (300, 300)
            im.thumbnail(size, Image.ANTIALIAS)
        
        all_faces.append(np.array(im.convert('L')))
        
        im_raw = np.asarray(im)
        irow, icol = im_raw.shape
        
        # Reshape and add to the main matrix
        
        temp = np.reshape(im_raw, irow*icol, order='C')
        S.append(temp)
    
    return S, all_faces
    
def gray_images_square_grid(images, mode='L'):
    """
    Save images as a square grid
    :param images: Images to be used for the grid
    :param mode: The mode to use for images
    :return: Image of images in a square grid
    """
    # Get maximum size for square grid of images
    save_size = math.floor(np.sqrt(images.shape[0]))

    # Put images in a square arrangement
    images_in_square = np.reshape(
            images[:save_size*save_size],
            (save_size, save_size, images.shape[1], images.shape[2]))

    # Combine images to grid image
    new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
    for col_i, col_images in enumerate(images_in_square):
        for image_i, image in enumerate(col_images):
            im = Image.fromarray(image, mode)
            new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

    return new_im

def normalizeImages(S):
    for i in range(len(S)):
        temp = S[i]
        m = np.mean(temp)
        st = np.std(temp)
        norm = (temp-m)*st/(st+m)
        S[i] = norm
        
    return S
        
def computeAverageFace(S):
    # Convert S to appropriate matrix form
    
    S_f = np.array(S)
    m = np.mean(S_f, axis = 0)
    
    return m
    
        