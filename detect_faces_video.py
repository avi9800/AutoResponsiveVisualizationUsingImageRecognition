
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
        help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

image=cv2.imread("hello.jpg")
h,w,c=image.shape
h1,w1=h,w

prototxt='deploy.prototxt.txt'
model='res10_300x300_ssd_iter_140000.caffemodel'



print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt,model)


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
resize_factor=10
c=0
area=0
constant_height,constant_width=0,0
constant_h,constant_w=1,1

def image_display(new_area,constant_height,constant_width):
        new_height=constant_height*new_area
        new_width=constant_width*new_area
        return int(new_height),int(new_width)
        

while True:
        
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
 
        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
 
        
        net.setInput(blob)
        detections = net.forward()

        
        for i in range(0, detections.shape[2]):
                
                confidence = detections[0, 0, i, 2]

                
                if confidence < args["confidence"]:
                        continue

                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                area=((startX-endX)*(startY-endY))
                diff=area-c
                
                if area<0:
                        area=area*-1
                
                if diff>2000 or diff<(-2000):
                        if (startX-endX)-(startY-endY)>10:
                                c=area
                area=c
                
                text = "{:.2f}%".format(confidence * 100)
                text=text+" "+"Area= "+str(area)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("i"):
                h1=h1+resize_factor
                w1=w1+resize_factor
                img=cv2.resize(image,(w1,h1))
                cv2.imshow('l',img)
                print(h1,w1,area)
                
        if key == ord("o"):
                h1=h1-resize_factor
                w1=w1-resize_factor
                img=cv2.resize(image,(w1,h1))
                cv2.imshow('l',img)
 
        
        if key == ord("s"):
                if area != 0:
                        constant_h=(h1*area)
                        constant_w=(w1*area)
                else:
                        print("Face not detectable!!!")
                        continue

        if constant_h!=1 and constant_w!=1:
                h1=int(constant_h/area)
                w1=int(constant_w/area)

        if key == ord("p"):
                print(h1,w1)
                
        img=cv2.resize(image,(w1,h1))
        cv2.imshow('l',img)
                
        if key == ord("q"):
                break




cv2.destroyAllWindows()
vs.stop()
