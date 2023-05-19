# Import required packages:
import cv2
import dlib
import numpy as np
import imutils

from utils import * 

def image_capture(detector, predictor, filename):
    # Capture frame from the VideoCapture object:
    test_face = cv2.imread(filename)

    # Just for debugging purposes:
    frame = test_face.copy()

    # Convert frame to grayscale:
    #frame = cv2.resize(frame,(0,0),fx = 0.5 , fy = 0.5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    

    # Detect faces:
    rects = detector(gray, 0)

    points_rights = []
    points_lefts = []

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(rects):
        # Draw a box around the face:
        #cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 1)

        # Get the shape using the predictor:
        shape = predictor(gray, rect)

        # Convert the shape to numpy array:
        shape = shape_to_np(shape)

        #Draw all lines connecting the different face parts:
        #draw_shape_lines_all(shape, frame)
        #draw_shape_lines_range(shape, frame, range_points, is_closed=False)

        draw_shape_lines_range(shape, frame, LEFT_EYEBROW_POINTS)
        draw_shape_lines_range(shape, frame, RIGHT_EYEBROW_POINTS)


        
        # right eyebrow 
        (x, y, w, h) = cv2.boundingRect(np.array([shape[RIGHT_EYEBROW_POINTS]]))
        roi = frame[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("ROI_right", roi)

        roi_right = roi 
        points_right = shape[RIGHT_EYEBROW_POINTS]

        points_rights.append(points_right)




        # left eyebrow 
        (x, y, w, h) = cv2.boundingRect(np.array([shape[LEFT_EYEBROW_POINTS]]))
        roi = frame[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("ROI_left", roi)

        roi_left = roi 
        points_left = shape[LEFT_EYEBROW_POINTS]

        points_lefts.append(points_left)

        translate_eyebrow(points_right, points_left, frame, is_closed = False)


        # Draw jaw line:
        #draw_shape_lines_range(shape, frame, JAWLINE_POINTS)

        # Draw all points and their position:
        #draw_shape_points_pos(shape, frame)
        # You can also use:
        # draw_shape_points_pos_range(shape, frame, ALL_POINTS)

        # Draw all shape points:
        #draw_shape_points(shape, frame)

        # Draw left eye, right eye and bridge shape points and positions
        # draw_shape_points_pos_range(shape, frame, LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_BRIDGE_POINTS)

        # Display the resulting frame
        #out.write(frame)
        cv2.imshow("Landmarks detection using dlib", frame)

        # Press 'q' key to exit
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    return roi_right, roi_left, points_rights, points_lefts