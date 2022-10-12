import cv2
import matplotlib.pyplot as plt

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model,config_file)

# opening the file
file_obj = open("label.txt", "r")
  
# reading the data from the file
file_data = file_obj.read()
  
# splitting the file data into lines
labels = file_data.splitlines()
print(labels)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ##255/2
model.setInputMean((127.5,127.5,127.5)) ## mobilenet => [-1,1]
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)

#check video
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('Cant open video')
    
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            cv2.rectangle(frame, boxes,(255,0,0),2)
            cv2.putText(frame, labels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)
        
    cv2.imshow('Object Detection', frame)
    
    if cv2.waitKey(2) & 0xFF ==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()