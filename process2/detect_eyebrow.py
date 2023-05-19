# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

image_filename = "face1.png"
model_weight_filename = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_weight_filename)


image = cv2.imread(image_filename)
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


rects = detector(gray, 1)

clone = image.copy()
# loop over the face detections
for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
	#for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
	# clone the original image so we can draw on it, then
	# display the name of the face part on the image
	#clone = image.copy()
	#cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	#	0.7, (0, 0, 255), 2)
	# loop over the subset of facial landmarks, drawing the
	# specific face part
	for (x, y) in shape[17:27]:
		cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
cv2.imshow("Image", clone)
cv2.waitKey(0)