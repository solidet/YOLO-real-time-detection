#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 18:49:21 2019

@author: solidet
"""

import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load Image
#
# img = cv2.imread("yolo_object_detection/me.jpg")
# img = cv2.resize(img, None, fx=1, fy=1)
# height, width, channels = img.shape

#load camera
cap = cv2.VideoCapture(0)

starting_time = time.time()
frame_id = 0

while True:
    _, frame = cap.read()
    print(frame)
    frame_id += 1
    height, width, channels = frame.shape

    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# for b in blob:
#     for name, img_blob in enumerate(b):
#         cv2.imshow(str(name), img_blob)

    net.setInput(blob)
    outs = net.forward(output_layers)


    # showing info on the screen
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

                # cv2.circle(img, (center_x, center_y), 10, (0, 255), 2)

                # Rectangle coordinates
                y = int(center_y - h / 2)
                x = int(center_x - w / 2)
                boxes.append([x, y, w, h])
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
            label = str(classes[class_ids[i]]) + ":" + str(format(confidences[i] * 100, '.0f')) + "%"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y + 20), font, 0.5, (0, 0, 255), 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()