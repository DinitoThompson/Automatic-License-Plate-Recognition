# Import all the needed libraries
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
from Utility.util import BoxUtil
from datetime import datetime

boxUtil = BoxUtil()

# Constants
model_cfg_path = 'Utility\Model\cfg\darknet-yolov3.cfg'
model_weights_path = 'Utility\Model\weights\model.weights'
save_path = "Data/Saved_Plates"


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

        path_name = f"{save_path}/{file_name}"

        resized_license_plate = cv2.resize(
            license_plate, None, fx=3.0, fy=3.0)

        return cv2.imwrite(path_name, resized_license_plate)

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

        plate_output = reader.readtext(plate)
        gray_output = reader.readtext(gray)
        thresh_output = reader.readtext(thresh)

        highest_score = 0.0
        text = ""
        lp_used = ""

        for out in plate_output:
            text_bbox, text, text_score = out
            highest_score = text_score
            text = text
            lp_used = "License Plate (Colored)"

        for out in gray_output:
            text_bbox, text, text_score = out
            if text_score > highest_score:
                highest_score = text_score
                text = text
                lp_used = "License Plate (Gray)"

        for out in thresh_output:
            text_bbox, text, text_score = out
            if text_score > highest_score:
                highest_score = text_score
                text = text
                lp_used = "License Plate (Edged)"

        return highest_score, text, lp_used
