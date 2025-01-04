import os
from pathlib import Path
import xml.etree.ElementTree as ET
from cv2 import imread
from dataclasses import dataclass, field
from typing import List
import numpy as np

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
    image: np.ndarray = None
    objects: List[ObjectAnnotation] = field(default_factory=list)



#  Current Path
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Project Root
# project_root = os.path.join(current_dir, "..", "..")

# # Construct paths to other directories
# data_dir = os.path.join(project_root, "data")
# raw_data_dir = os.path.join(project_root, "raw")

def get_data_sample(image_path: Path, annotation_path: Path) -> ImageData:
    """
    Parses an image and its annotation file to create a data sample

    Args:
        image_path: Path to image file
        annotation_path: Path to the XML annotation file

    Returns:
        An ImageData object containing the parsed data.

    Raises:
        FileNotFoundError: If the image or annotation file does not exist.
        ValueError: If there is an error parsing the annotation file.
    """
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    

    try:   
        tree = ET.parse(str(annotation_path))
        root = tree.getroot()

        filename = root.find("filename").text
        filepath = str(image_path)
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        image = imread(filepath)
        objects = []
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            objects.append(ObjectAnnotation(class_name, BoundingBox(xmin, ymin, xmax, ymax)))

        return ImageData(filename, filepath, width, height, image, objects)
    
    except (ValueError, ET.ParseError) as e:
        raise ValueError(f"Error parsing annotation file: {annotation_path} - {e}")