# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:15:28 2019
@author: anishukla
"""

# Libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# Defining a function that will do detection
"""There will be a for loop for each an every faces detected in the video.
 Importantly, we will do the eye detection part inside the face detection part
 of the video"""
 
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    "We'll get coordinates of faces"
    "Above given object gives coordinates of rectangle around face."
    # detectMultiScale(color,scaling factor, neighbours)
    # 1.3 and 5 are the experimental and optmum values for webcam.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #it has parameter frame, coordinates of bottom right and top left 
        #then we give rgb code for color of rectangle and width of rectagle.
        roi_gray = gray[y:y+h, x:x+w]
        #We need region of interest for both gray and colored image
        roi_color = faces[y:y+h, x:x+w]
         
         
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # The region here will be face as it reduces the area of detection.
        # and obviously eyes are inside the face.
        for (ex, ey, ew, eh) in eyes:   
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1.5)
            "Rectangle around eyes."
        
    return frame

# Doing face recog with webcam
video_capture = cv2.VideoCapture(0)
"video_capture class"
# 0 for internal webcam and 1 for external
while True:
    _, frame = video_capture.read()
    #video_capture.read() object gives us 2 outputs so we
    # kept 1 in _ not needed and other as frames.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    "Now we need to display all successive outputs in an animated way."
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
