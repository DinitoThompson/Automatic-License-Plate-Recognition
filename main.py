# Import files
from Utility.detector import Detector
from Blacklist import *

# Sam Semegment Anything Model
# Microsoft Azure - Cognative Services

# Read different file type
imagePath = "./Data/Images/Vehicle (1).png"
# webcamPath = 0

# Initialize Detector Class
detector = Detector()

# Read image
detector.LP_Image_Detection(imagePath)

# Read video

# if __name__ == '__main__':
#
#    Blacklist.addToBlacklist("5")
