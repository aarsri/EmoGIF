# Author: Aarohi Srivastava

import sys, cv2, dlib, time, os, math
import numpy as np
import cv2, dlib, os
import mls as mls
import numpy as np
import math
import argparse
import sys
from ProjectPaintingLight import run, getFrames, min_resize
import random
from zenipy import entry, file_selection, message

# dictionary to look up the appropriate driver video, relighting scheme, and filter based on the user-inputted emotion
emotions_dict = {
    'excited': {'driver': 'big_smile.mp4', 'relighting': 'sporadic', 'filter': 'warm'},
    'happy': {'driver': 'big_smile.mp4', 'relighting': 'circle', 'filter': 'warm'},
    'whimsical': {'driver': 'big_smile.mp4', 'relighting': 'circle', 'filter': 'cartoon'},
    'love': {'driver': 'soft_smile.mp4', 'relighting': 'circle', 'filter': 'pink'},
    'peaceful': {'driver': 'soft_smile.mp4', 'relighting': 'vertical', 'filter': 'blue'},
    'nostalgic': {'driver': 'soft_smile.mp4', 'relighting': 'diagonal', 'filter': 'sepia'},
    'gloomy': {'driver': 'pout.mp4', 'relighting': 'bottom', 'filter': 'orange'},
    'confused': {'driver': 'furrow.mp4', 'relighting': 'sporadic', 'filter': 'orange'},
    'cutoff': {'driver': 'cutoff.mp4', 'relighting': 'diagonal', 'filter': 'sepia'}, #to test out cutoff driver video
    'tilt': {'driver': 'tilt.mp4', 'relighting': 'diagonal', 'filter': 'sepia'} #to test out tilted driver video
}

# get list of positions for relighting scheme
def getPositionsList(emotion, fps):
    if emotions_dict[emotion]['relighting']=='vertical':
        ys = list(np.linspace(1, -1, num=fps)) #array of descending y coordinates
        return [(0, y) for y in ys] #x fixed at 0 to go down the middle
    elif emotions_dict[emotion]['relighting']=='sporadic':
        return [(random.uniform(-1,1), random.uniform(-1,1)) for i in range(fps)] #array of random x y coordinates
    elif emotions_dict[emotion]['relighting']=='circle':
        thetas = list(np.linspace(0, 2*math.pi, num=fps)) #array of angles around a circle
        thetas.reverse() #clockwise
        r = 0.7 #radius 
        return [(r*math.sin(theta), r*math.cos(theta)) for theta in thetas] #list of points about a circle of radius 0.7
    elif emotions_dict[emotion]['relighting']=='diagonal':
        xs = list(np.linspace(1, -1, num=fps)) #array of decreasing x coordinates
        ys = list(np.linspace(-1, 1, num=fps)) #array of increasing y coordinates
        return [(x, y) for x, y in zip(xs, ys)] #list of points across the diagonals
    elif emotions_dict[emotion]['relighting']=='bottom':
        xs = list(np.linspace(1, -1, num=fps)) #array of x coordinates
        return [(x, -0.75) for x in xs] #y fixed at -0.75 to go across the bottom
    else:
        return []
 
# relight the first frame - requires additional computations 
def firstFrameRelighting(image, emotion, positions_list):
    if emotions_dict[emotion]['relighting'] == 'none':
        return image
    light_color_red = 1.0
    light_color_green = 1.0
    light_color_blue = 1.0
    # parameters tuned by Aarohi Srivastava for Relighting:
	    # ambient intensity = 0.6
	    # light intensity = 0.85
	    # light source height = 0.85
    rendered_image, light_source_color, index_tri, index_ray, locations = run(image, None, 0.6, 0.85, 0.85, 1.0, 1.2, light_color_red, light_color_green, light_color_blue, True, positions_list[0][0], positions_list[0][1])
    return rendered_image, light_source_color, index_tri, index_ray, locations
    #params: image, mask, ambient_intensity, light_intensity, light_source_height,gamma_correction, stroke_density_clipping, light_color_red, light_color_green,light_color_blue, enabling_multiple_channel_effects

# apply filter to frame 
def getFullFiltered(img, emotion):
    if emotions_dict[emotion]['filter'] == 'sepia':
        img = np.array(img, dtype=np.float64)
        img = cv2.transform(img, np.matrix([[0.272, 0.534, 0.131],
                                            [0.349, 0.686, 0.168],
                                            [0.393, 0.769, 0.189]])) #sepia matrix

    elif emotions_dict[emotion]['filter'] == 'blue':
        blue = np.full(img.shape, (252,177,3), np.uint8) #rgb values for blue tint
        img = 0.15*blue + 0.85*img #low opacity on tint
    elif emotions_dict[emotion]['filter'] == 'pink':
        pink = np.full(img.shape, (198, 161, 255), np.uint8) #rgb values for pink tint
        img = 0.15*pink + 0.85*img #low opacity on tint
    elif emotions_dict[emotion]['filter'] == 'orange':
        orange = np.full(img.shape, (98, 152, 245), np.uint8) #rgb values for orange tint
        img = 0.15*orange + 0.85*img - 30 #low opacity on tint; darken
    elif emotions_dict[emotion]['filter'] == 'warm':
        orange = np.full(img.shape, (0, 213, 255), np.uint8) #rgb values for warm tint
        img = 0.15*orange + 0.85*img + 10 #low opacity on tint; brighten
    elif emotions_dict[emotion]['filter'] == 'edges':
        img = cv2.Canny(img,100,400) # Canny edge detection
    elif emotions_dict[emotion]['filter'] == 'cartoon':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5) # applying median blur with kernel size of 5
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7) # Canny edge detection
        dst = cv2.edgePreservingFilter(img, flags=2, sigma_s=64, sigma_r=0.25)
        cartoon = cv2.bitwise_and(dst, dst, mask=edges) # adding thick edges to smoothened image
        img = cartoon

    return img.clip(0, 255).astype(np.uint8)

# apply filter to frame with alpha blending wrt frameCounter
def applyFilter(img, emotion, frameCounter, numFrames):
    full = getFullFiltered(img, emotion)
    if emotions_dict[emotion]['filter'] == 'cartoon': #no blending for cartoon filter
        return full
    weight = frameCounter/(numFrames-1)
    dst = cv2.addWeighted(full, weight, img, 1-weight, 0.0).clip(0, 255).astype(np.uint8) #alpha blending
    blur = cv2.bilateralFilter(dst,3,20,20)
    return blur
