import os
import xml.etree.ElementTree as ET
from cv2 import imread

# Current Path
current_dir = os.path.dirname(os.path.abspath(__file__))

# Project Root
project_root = os.path.join(current_dir, "..", "..")

# Construct paths to other directories
data_dir = os.path.join(project_root, "data")
raw_data_dir = os.path.join(project_root, "raw")