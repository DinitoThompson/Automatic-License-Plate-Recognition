# Import files
from Detector import *

# Read different file type
videoPath = ""
imagePath = "./Images/Vehicle (11).png"
webcamPath = 0

# Initialize Detector Class
detector = Detector()

# Read image
detector.imageDetection(imagePath)

# Read video
