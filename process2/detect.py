#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:11:40 2020
@author: pavankunchala
"""
# Import required packages:
import cv2
import dlib
import numpy as np
import imutils

from utils import * 
from from_camera import camera_capture
from from_video import video_capture
from from_image import image_capture


# Name of the two shape predictors:
p = "shape_predictor_68_face_landmarks.dat"
# p = "shape_predictor_5_face_landmarks.dat"

# Initialize frontal face detector and shape predictor:
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

input_type = str(input("Please enter input type (webcam, video, image): "))

if input_type == "webcam":
    camera_capture(detector, predictor)
    
elif input_type == "video":
    file_name = str(input("Please enter file name: "))
    video_capture(detector, predictor, file_name)

elif input_type == "image":
    file_name = str(input("Please enter file name: "))
    roi_right, roi_left, points_right, points_left = image_capture(detector, predictor, file_name)

    #print("points_right: ", points_right)
    #print("points_left: ", points_left)
else: 
    print("value error")


