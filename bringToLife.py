#!/usr/bin/env python
# coding: utf-8

# In[1]:

#inspired from https://github.com/ekkravchenko/livePortraits (with modification)

# import necessary packages and functions
import cv2, dlib, os
import mls as mls
import numpy as np
import math
from utils import getFaceRect, landmarks2numpy, createSubdiv2D, calculateDelaunayTriangles, insertBoundaryPoints, getVideoParameters, warpTriangle, getRigidAlignment, teethMaskCreate, erodeLipMask, getLips, getLipHeight, drawDelaunay, mainWarpField, copyMouth, hallucinateControlPoints, getInterEyeDistance
from aarohi_utils import emotions_dict, getPositionsList, firstFrameRelighting, getFullFiltered, applyFilter # code by Aarohi Srivastava
import argparse
import sys
from ProjectPaintingLight import run, getFrames, min_resize
import random
from zenipy import entry, file_selection, message

assert cv2.__version__ == '4.5.0'

# create a folder for results, if it doesn't exist yet
os.makedirs("video_generated", exist_ok=True) 

""" Get face and landmark detectors"""
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Landmark model location
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)  


def startCap(im_fp, emotion, output_fn):
    video_fn = emotions_dict[emotion]['driver'] #obtain file name of driver video based on input emotion 
    cap = cv2.VideoCapture(os.path.join("driver_videos", video_fn))
    if (cap.isOpened() == False): 
        print("Unable to read video")
    im = cv2.imread(im_fp)
    if im is None:
        print("Unable to read photo")
    else:
        im = min_resize(im, 512)
    runLivePortraits(im_fp, im, video_fn, output_fn, cap, emotion)

""" Main algorithm """
def runLivePortraits(im_fp, im, video_fn, output_fn, cap, emotion):
    
    # param initialization
    im_height, im_width, im_channels = im.shape
    (time_video, length_video, fps, frame_width, frame_height) = getVideoParameters(cap)
    # detect the face and the landmarks
    newRect = getFaceRect(im, faceDetector)
    landmarks_im = landmarks2numpy(landmarkDetector(im, newRect))

    # create output video at double fps; goes into '/video_generated/' directory 
    out = cv2.VideoWriter(os.path.join("video_generated", output_fn), cv2.VideoWriter_fourcc(*'mp4v'), fps*2, (int(im_width/2), int(im_height/2)))

    frame = [] 
    tform = [] # similarity transformation that alignes video frame to the input image
    srcPoints_frame = []
    numCP = 68 # number of control points
    newRect_frame = []

    # Optical Flow
    points=[]
    pointsPrev=[] 
    pointsDetectedCur=[] 
    pointsDetectedPrev=[]
    eyeDistanceNotCalculated = True
    eyeDistance = 0
    isFirstFrame = True
    frameCounter = 0
    numFrames = length_video

    # Get relighting positions list -- new from Aarohi Srivastava
    positions_list = getPositionsList(emotion, numFrames)

   #frame-wise processing
    while True:
        ret, frame = cap.read()
        if ret == False:
            break  
        else: ############### Optical Flow and Stabilization #######################
            # initialize a new frame for the input image
            im_new = im.copy()          
            frame = min_resize(frame, 512)
            
            # detect the face (only for the first frame) and landmarks
            if isFirstFrame: 
                newRect_frame = getFaceRect(frame, faceDetector)
                landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
                # compute the similarity transformation in the first frame
                tform = getRigidAlignment(landmarks_frame_init, landmarks_im)    
            else:
                landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
                if len(tform) == 0:
                    print("ERROR: NO SIMILARITY TRANSFORMATION")

            # Apply similarity transform to the frame
            frame_aligned = np.zeros((im_height, im_width, im_channels), dtype=im.dtype)
            frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))

            # Change the landmarks locations
            landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
            landmarks_frame = cv2.transform(landmarks_frame, tform)
            landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))

            # hallucinate additional control points
            if isFirstFrame: 
                (subdiv_temp, dt_im, landmarks_frame) = hallucinateControlPoints(landmarks_init = landmarks_frame, 
                                                                                im_shape = frame_aligned.shape, 
                                                                                INPUT_DIR="", 
                                                                                performTriangulation = True)
                # number of control points
                numCP = landmarks_frame.shape[0]
            else:
                landmarks_frame = np.concatenate((landmarks_frame, np.zeros((numCP-68,2))), axis=0)

            ############### Optical Flow and Stabilization #######################
            # Convert to grayscale.
            imGray = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2GRAY)

            # prepare data for an optical flow
            if (isFirstFrame==True):
                [pointsPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
                [pointsDetectedPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
                imGrayPrev = imGray.copy()

            # pointsDetectedCur stores results returned by the facial landmark detector
            # points stores the stabilized landmark points
            points = []
            pointsDetectedCur = []
            [points.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            [pointsDetectedCur.append((p[0], p[1])) for p in landmarks_frame[68:,:]]

            # Convert to numpy float array
            pointsArr = np.array(points, np.float32)
            pointsPrevArr = np.array(pointsPrev,np.float32)

            # If eye distance is not calculated before
            if eyeDistanceNotCalculated:
                eyeDistance = getInterEyeDistance(landmarks_frame)
                eyeDistanceNotCalculated = False

            dotRadius = 3 if (eyeDistance > 100) else 2
            sigma = eyeDistance * eyeDistance / 400
            s = 2*int(eyeDistance/4)+1

            #  Set up optical flow params
            lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
            pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev,imGray,pointsPrevArr,pointsArr,**lk_params)
            sigma = 100

            # Converting to float and back to list
            points = np.array(pointsArr,np.float32).tolist()   

            # Facial landmark points are the detected landmark and additional control points are tracked landmarks  
            landmarks_frame[68:,:] = pointsArr
            landmarks_frame = landmarks_frame.astype(np.int32)

            # getting ready for the next frame
            imGrayPrev = imGray        
            pointsPrev = points
            pointsDetectedPrev = pointsDetectedCur

            ############### End of Optical Flow and Stabilization #######################

            # save information of the first frame for the future
            if isFirstFrame: 
                # hallucinate additional control points for a still image
                landmarks_list = landmarks_im.copy().tolist()
                for p in landmarks_frame[68:]:
                    landmarks_list.append([p[0], p[1]])
                srcPoints = np.array(landmarks_list)
                srcPoints = insertBoundaryPoints(im_width, im_height, srcPoints) 

                lip_height = getLipHeight(landmarks_im)            
                (_, _, maskInnerLips0, _) = teethMaskCreate(im_height, im_width, srcPoints)    
                mouth_area0=maskInnerLips0.sum()/255  

                # get source location on the first frame
                srcPoints_frame = landmarks_frame.copy()
                srcPoints_frame = insertBoundaryPoints(im_width, im_height, srcPoints_frame)  

                # Write the original image into the output file
                if emotions_dict[emotion]['relighting'] != 'none':
                    im_new, light_source_color, index_tri, index_ray, locations = firstFrameRelighting(im_new, emotion, positions_list)
                
                # filter
                im_new = applyFilter(im_new, emotion, frameCounter, numFrames)

                out.write(im_new)
                isFirstFrame = False
                print("Frame-wise processing...")
                continue

            ############### Warp field #######################               
            dstPoints_frame = landmarks_frame
            dstPoints_frame = insertBoundaryPoints(im_width, im_height, dstPoints_frame)

            # get the new locations of the control points
            dstPoints = dstPoints_frame - srcPoints_frame + srcPoints   

            # get a warp field, smoothen it and warp the image
            im_new = mainWarpField(im,srcPoints,dstPoints,dt_im)       

            ############### Mouth cloning #######################
            # get the lips and teeth mask
            (maskAllLips, hullOuterLipsIndex, maskInnerLips, hullInnerLipsIndex) = teethMaskCreate(im_height, im_width, dstPoints)
            mouth_area = maskInnerLips.sum()/255        

            # erode the outer mask based on lipHeight
            maskAllLipsEroded = erodeLipMask(maskAllLips, lip_height)
            
            # smooth the mask of inner region of the mouth
            maskInnerLips = cv2.GaussianBlur(np.stack((maskInnerLips,maskInnerLips,maskInnerLips), axis=2),(3,3), 10)

            # clone/blend the moth part from 'frame_aligned' if needed (for mouth_area/mouth_area0 > 1)
            im_new = copyMouth(mouth_area, mouth_area0,
                                landmarks_frame, dstPoints,
                                frame_aligned, im_new,
                                maskAllLipsEroded, hullOuterLipsIndex, maskInnerLips)           

            # relighting -- new from Aarohi Srivastava
            if emotions_dict[emotion]['relighting'] != 'none':
                # parameters tuned by Aarohi Srivastava for Relighting:
                    # ambient intensity = 0.6
                    # light intensity = 0.85
                    # light source height = 0.85
                im_new = getFrames(im_new, None, 0.6, 0.85, 1.0, 1.0, 1.2, True, positions_list[frameCounter][0], positions_list[frameCounter][1], light_source_color, index_tri, index_ray, locations)

            # filter -- new from Aarohi Srivastava
            im_new = applyFilter(im_new, emotion, frameCounter, numFrames)

            # Write the frame into the file 'output.avi'
            out.write(im_new)
            frameCounter += 1
                           
    # When everything is done, release the video capture and video write objects
    out.release()
    print("released")
    cap.release()


if __name__ == '__main__': # Aarohi Srivastava
    args = sys.argv
    im_fp = args[1] #example: 'input_images/Frida1.jpg' 
    emotion = args[2] #example: 'peaceful'
    output_fn = args[3] #example: 'Frida1-peaceful.mp4'
    startCap(im_fp, emotion, output_fn) #open video capture object and start
    exit()
