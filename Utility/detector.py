# Import all the needed libraries
import os
import imutils
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
harcascade = "Utility\Model\Haarcascade\haarcascade_russian_plate_number.xml"
save_path = "Data/Saved_Plates"
feed_save_path = "Data/Saved_Plates_Feed"
reader = easyocr.Reader(['en'])

# Load model
yoloModel = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)


class Detector:
    def __init__(self) -> None:
        pass

# -----------------------IMAGE DETECTION-----------------------#

    def LP_Image_Detection(self, imagePath):
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

            # Saves detected license plate
            # if (self.LP_Saver(license_plate, results[1])):
            #     print("License Plate Saved.")

            self.LP_Filter_Status(results)

            # Displays detected results
            self.LP_Results(image, license_plate,
                            license_plate_gray, license_plate_edged)

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

    def LP_Reader(self, plate):
        result = reader.readtext(plate)
        return result

    def LP_Filter(self, plate, gray, thresh):

        plate_output = self.LP_Reader(plate)
        gray_output = self.LP_Reader(gray)
        thresh_output = self.LP_Reader(thresh)

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

# -----------------------LIVE FEED-----------------------#

    def LP_Live_Feed(self, webcamPath):
        cap = cv2.VideoCapture(webcamPath)

        if (cap.isOpened() == False):
            print("Error Accessing Webcam: " + webcamPath)
            return

        while True:
            success, frame = cap.read()
            if success:
                cv2.imshow("Webcam Feed", frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
        print(f"[INFO] Live Feed Closed . . . ")
        # closing all windows
        cv2.destroyAllWindows()
        return

# -----------------------RESERVED PARKING LOT FEED-----------------------#

    def LP_Parking_Lot_Feed(self, videoPath):

        plate_cascade = cv2.CascadeClassifier(harcascade)
        cap = cv2.VideoCapture(videoPath)

        # cap.set(3, 340)  # width
        # cap.set(4, 280)  # height

        min_area = 400
        count = 0

        if (cap.isOpened() == False):
            print("Error Accessing Video: " + videoPath)
            return

        while True:
            success, frame = cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            plates = plate_cascade.detectMultiScale(frame_gray, 2.5, 4)

            for (x, y, w, h) in plates:
                area = w * h

                if area > min_area:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Plate Detected", (x, y-5),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

                    img_roi = frame[y: y+h, x:x+w]

                    cv2.imshow(f"License Plate", img_roi)

            cv2.imshow("Result", frame)

            # TODO: Make this more efficent
            if cv2.waitKey(1) & 0xFF == ord('s'):
                file_name = datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + ".png"
                path_name = f"{feed_save_path}/{file_name}"
                cv2.imwrite(path_name, img_roi)
                cv2.rectangle(frame, (0, 200), (640, 300),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Plate Saved", (150, 265),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
                cv2.imshow("Results", frame)
                cv2.waitKey(500)
                count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print(f"[INFO] Live Feed Closed . . . ")
        # closing all windows
        cv2.destroyAllWindows()
        return
