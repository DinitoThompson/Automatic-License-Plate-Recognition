# Import files
from Detector import *

# Read different file type
videoPath = "./Videos/Video (3).mp4"
imagePath = "./Images/JA_Vehicles (7).jpg"
webcamPath = 0

# Initialize Detector Class
detector = Detector()

# Read image
# detector.LP_Image_Detection(imagePath)

# Read video
detector.LP_Video_Detection_2(videoPath)
