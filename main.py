# Import files
from Utility.detector import Detector

# Sam Semegment Anything Model
# Microsoft Azure - Cognative Services

# Read different file type
imagePath = "./Data/Images/Vehicle (1).png"
videoPath = "./Data/Videos/Video (3).mp4"
webcamPath = 0

# Initialize Detector Class
detector = Detector()

# Read Image
# detector.LP_Image_Detection(imagePath)

# Webcam Feed
# detector.LP_Live_Feed(webcamPath)

# Parking Lot Feed
# detector.LP_Parking_Lot_Feed(videoPath)
