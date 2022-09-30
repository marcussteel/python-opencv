import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model,config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

classLabels = []
file_name = "src/recognizers/cocolabels.txt"

with open(file_name, "rt") as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

#print(classLabels)
#print(len(classLabels))

#----------------------Read An Image ------------------------
# if you look at coco config image size 320

# model.setInputSize(320, 320)
# model.setInputScale(1.0 / 127.5)
# model.setInputMean((127.5, 127.5, 127.5))
# model.setInputSwapRB(True)

# img = cv2.imread("src/images/kit-harington/1.jpg")
# plt.imshow(img)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)
    #print(ClassIndex)

    if len(ClassIndex!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0,255,0), thickness=3)
    
    # plt.imshow(img)

  # Display the resulting frame
    cv2.imshow('Object Detection Program', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
