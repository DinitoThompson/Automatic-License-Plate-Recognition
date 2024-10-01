# Import all the needed libraries
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from Utility.util import BoxUtil
from datetime import datetime
from black_list import *


boxUtil = BoxUtil()

# Constants
model_cfg_path = 'Utility\Model\cfg\darknet-yolov3.cfg'
model_weights_path = 'Utility\Model\weights\model.weights'
harcascade = "Utility\Model\Haarcascade\haarcascade_russian_plate_number.xml"
save_path = "Data/Saved_Plates"
feed_save_path = "Data/Saved_Plates_Feed"
reader = easyocr.Reader(['en'])
blacklist_path = "Data/Blacklist_Plates"

# Load model
yoloModel = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)


class Detector:
    def __init__(self) -> None:
        pass

    def LP_Image_Detection(self, imagePath):
        # Load model
        yoloModel = cv2.dnn.readNetFromDarknet(
            model_cfg_path, model_weights_path)

        # Load image
        image = cv2.imread(imagePath)

        # Get Height & Width
        H, W, _ = image.shape

        # Convert image to 4D Blob
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255, (416, 416), (0, 0, 0), True)

        # Get license plate detections from blob
        yoloModel.setInput(blob)

        # Extract the detections
        detections = boxUtil.get_outputs(yoloModel)

        # Apply nms
        bboxes, class_ids, scores = self.LP_Plate_Detection(W, H, detections)

        # Plot region of interest
        for bbox_, bbox in enumerate(bboxes):

            license_plate, image = self.LP_Plot_Region(image, bbox)

            license_plate_gray = cv2.cvtColor(
                license_plate, cv2.COLOR_BGR2GRAY)

            _, license_plate_edged = cv2.threshold(
                license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

            results = self.LP_Filter(
                license_plate, license_plate_gray, license_plate_edged)

            if (self.LP_Saver(license_plate, results[1])):
                print("License Plate Saved.")

            self.LP_Filter_Status(results)

            # self.LP_Results(image, license_plate,
            #                 license_plate_gray, license_plate_edged)

        plt.show()

        return

    def LP_Plot_Region(self, image, bbox):
        xc, yc, w, h = bbox

        cv2.putText(image,
                    "License Plate",
                    (int(xc - (w / 2)) + 30, int(yc + (h / 2) + 55)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    4)

        license_plate = image[int(yc - (h / 2)):int(yc + (h / 2)),
                              int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

        image = cv2.rectangle(image,
                              (int(xc - (w / 2)), int(yc - (h / 2))),
                              (int(xc + (w / 2)), int(yc + (h / 2))),
                              (0, 255, 0),
                              thickness=5)

        return license_plate, image

    def LP_Saver(self, license_plate, license_plate_text):
        # 1-1-2000_01:00:00_PGJA34
        # Date(Month-Day-Year)_Time(Hour-Minute-Second)_License Plate

        file_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + "_" + \
            license_plate_text.replace(" ", "") + ".png"

        save_name = f"{save_path}/{file_name}"
        blacklist = f"{blacklist_path}/{file_name}"

        resized_license_plate = cv2.resize(
            license_plate, None, fx=3.0, fy=3.0)

        if (Blacklist.autoCheckBlacklist(license_plate_text)):
            return cv2.imwrite(save_name, resized_license_plate)
        else:
            return cv2.imwrite(blacklist, resized_license_plate)

    def LP_Filter_Status(self, results):
        print("License Plate: ", results[1])
        print("Confidence Value: %", round((results[0] * 100), 2))
        print("Image Used: ", results[2])

    def LP_Results(self, image, license_plate, license_plate_gray, license_plate_edged):
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate_gray, cv2.COLOR_BGR2RGB))

        plt.figure()
        plt.imshow(cv2.cvtColor(license_plate_edged, cv2.COLOR_BGR2RGB))

    def LP_Plate_Detection(self, W, H, detections):
        # bboxes, class_ids, confidences
        bboxes = []
        class_ids = []
        scores = []

        # Goes through and extract the detection with the score
        for detection in detections:
            # [x1, x2, x3, x4, x5, x6, ..., x85]
            bbox = detection[:4]

            xc, yc, w, h = bbox
            bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

            bbox_confidence = detection[4]

            class_id = np.argmax(detection[5:])
            score = np.amax(detection[5:])

            bboxes.append(bbox)
            class_ids.append(class_id)
            scores.append(score)

        return boxUtil.NMS(bboxes, class_ids, scores)

    def LP_Filter(self, plate, gray, thresh):

        reader = easyocr.Reader(['en'])

        outputArray = np.array([
            normalize_reader_output(reader.readtext(
                plate), "License Plate (Colored)"),
            normalize_reader_output(reader.readtext(
                gray), "License Plate (Gray)"),
            normalize_reader_output(reader.readtext(
                thresh), "License Plate (Edged)")
        ], dtype=object)

        maxScoreIndex = np.argmax(outputArray[:, 2])

        [_, text, score, lp_used] = outputArray[maxScoreIndex]

        return score, text, lp_used

def normalize_reader_output(output, label):
    # replace empty outputs with template
    if (len(output) == 0):
        normalized = [[[0, 0], [0, 0], [0, 0], [0, 0]], "", 0, label]
    else:
        # transform output tuple to list
        normalized = list(output[0])
        normalized.append(label)
    return normalized
