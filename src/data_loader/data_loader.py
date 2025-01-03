import os
import xml.etree.ElementTree as ET
from cv2 import imread
from dataclasses import dataclass, field
from typing import List

# Define data structure
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@dataclass
class ObjectAnnotation:
    class_name: str
    bbox: BoundingBox

@dataclass
class ImageData:
    filename: str
    filepath: str
    width: int
    height: int
    objects: List[ObjectAnnotation] = field(default_factory=list)



# # Current Path
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Project Root
# project_root = os.path.join(current_dir, "..", "..")

# # Construct paths to other directories
# data_dir = os.path.join(project_root, "data")
# raw_data_dir = os.path.join(project_root, "raw")