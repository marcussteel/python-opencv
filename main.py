import numpy as np
import cv2
from change_resolution import *
from change_color import change_color_to_grey
from record_video import *

cap = cv2.VideoCapture(0)

#to record the video
out = cv2.VideoWriter(filename, get_video_type(
    filename), 25, get_dims(cap, res))

# make_720p(cap) #change resolution.py ile


while True:
    
    # this is reading by frame so it is reading every frame
    ret, frame = cap.read()

    #to record the video
    out.write(frame)

    #change resolution.py ile frame i güncelleyebiliriz
    # frame = rescale_frame(frame, percent=70)

    # change_color_to_grey(cap)
      

    # now we are displayin this fram in our computer
    cv2.imshow('Frame : ', frame)  # imgshow

    #if you see grey bar and it just seemed to be hanging we should write this
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
