import cv2
import numpy as np
import os
import random
import imutils

class YOLOv3(object):
    def __init__(self, net, filename, layer_names, output_layers):
        self.net = net
        self.classes =  self.load_class_names(filename)
        self.layer_names = layer_names
        self.output_layers = output_layers
    
    def __call__(self, img):
        height, width, channels = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.05:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    end_x = x + w
                    end_y = y + h

                    boxes.append([x, y, end_x, end_y])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # for i in range(len(boxes)):
        #     if i in indexes:
        #         x, y, w, h = boxes[i]
        #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return boxes, confidences, class_ids

    def load_class_names(self, filename):
        with open(filename, 'r', encoding='utf8') as f:
            classes = [line.strip() for line in f.readlines()]
        
        return classes
    