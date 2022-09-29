import numpy as np
import cv2

print(cv2.__version__)

cap = cv2.VideoCapture(0)

while True:
    # this is reading by frame so it is reading every frame
    ret, frame = cap.read() 

    # now we are displayin this fram in our computer
    cv2.imshow('Frame : ', frame) #imgshow

    #we can put many frames :
    cv2.imshow('frame1 : ', frame) #imgshow

    #lets change color to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Frame With GRAY : ', gray)  # imgshow

    #if you see grey bar and it just seemed to be hanging we should write this
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

