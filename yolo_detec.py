#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:49:21 2019

@author: solidet
"""

import cv2
import numpy as np

#Load YOLO

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#Load Image

img = cv2.imread("./room.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4) # fx and fy values depend on the size of image
height, width, channels = img.shape

# detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


#showing info on the screen
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Center point
            # cv2.circle(img, (center_x, center_y), 10, (0, 255), 2)

            # Rectangle coordinates
            y = int(center_y - h / 2)
            x = int(center_x - w / 2)
            boxes.append([x,y, w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            # cv2.rectangle(img,(x,y),(x+w, y+h),(255, 0, 0), 2 )


indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(confidences)
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        print("class_ids:", class_ids[i])
        print("confidence:", confidences[i])
        label = str(classes[class_ids[i]]) + ":" + str(format(confidences[i]*100, '.0f')) + "%"
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, label, (x, y+20), font, 0.5, (0,0,255), 2)
cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()