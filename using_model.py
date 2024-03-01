import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader


class YOLO_pred():
    def __init__(self, onnx_model, data_yaml):
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, d), dtype=np.uint8)
        input_image[0: row, 0: col] = image
        
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        pred = self.yolo.forward()
        
        detections = pred[0]
        boxes = []
        confidences = []
        classes = []
        
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.5:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
        
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
        
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        
        #non maximum suppression
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
        if len(index) > 0:
            flattened = index.flatten()
            for ind in flattened:
                x, y, w, h = boxes_np[ind]
                bb_conf = confidences_np[ind]
                classes_id = classes[ind]
                class_name = self.labels[classes_id]

                text = f'{class_name}: {int(bb_conf * 100)}%'
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(image, (x, y + h - 30), (x + w, y + h), (255, 255, 255), -1)
                cv2.putText(image, text, (x, y + h - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
            
            #for counter
            if len(index) == 1:
                return image, class_name, bb_conf

        return image, '', 0
    
    def predictions_unprocessed(self, image):
        row, col, d = image.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, d), dtype=np.uint8)
        input_image[0: row, 0: col] = image
        
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        pred = self.yolo.forward()
        
        detections = pred[0]
        boxes = []
        confidences = []
        classes = []
        
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.5:
                class_score = row[5:].max()
                class_id = row[5:].argmax()
                if class_score > 0.25:
                    cx, cy, w, h = row[:4]
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
        
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
        
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        
        #non maximum suppression
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
        return boxes_np, confidences_np, classes, index