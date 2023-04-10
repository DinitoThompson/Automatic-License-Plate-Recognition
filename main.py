# Import files
from Utility.detector import Detector


# Read different file type
videoPath = ""
imagePath = "./Data/Images/JA_Vehicles (2).jpg"
webcamPath = 0

# Initialize Detector Class
detector = Detector()

# Read image
detector.LP_Image_Detection(imagePath)

# Read video
# detector.LP_Video_Detection_2(webcamPath)
# detector.LP_Video_Detection_3(videoPath)
