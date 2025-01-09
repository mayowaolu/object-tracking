import os
from pathlib import Path
import xml.etree.ElementTree as ET
from cv2 import imread
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional
import numpy as np
import logging
from tqdm import tqdm
import pickle

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



class DatasetPathError(Exception):
    """Custom exception for dataset path related errors"""
    pass


def create_data_splits(dataset_root: Union[str, Path]) -> Tuple[List[ImageData], List[ImageData]]:
    """
    Loads the Pascal VOC dataset, parses annotations, and creates training and testing data splits.

    Args:
        dataset_root: The root directory of the Pascal VOC dataset. (data/raw)

    Returns:
        A tuple containing two lists:
            - training_data: A list of ImageData objects for training.
            - testing_data: A list of ImageData objects for testing.
    """
    try:
        if not isinstance(dataset_root, (str, Path)):
            raise ValueError(f"Invalid type for dataset_root: {type(dataset_root)}. Expected str or Path.")
        dataset_root = Path(dataset_root).resolve()

        # Directory Structure
        paths = {
            'voc07': {
                'root': dataset_root / "VOC2007",
                'images': dataset_root / "VOC2007" / "JPEGImages",
                'annotations': dataset_root / "VOC2007" / "Annotations",
                'splits': dataset_root / "VOC2007" / "ImageSets" / "Main"
            },
            
            'voc12': {
                'root': dataset_root / "VOC2012",
                'images': dataset_root / "VOC2012" / "JPEGImages",
                'annotations': dataset_root / "VOC2012" / "Annotations",
                'splits': dataset_root / "VOC2012" / "ImageSets" / "Main"
            }
        }

        # Directory Validation
        for year, year_paths in paths.items():
            for path_type, path in year_paths.items():
                if not path.exists():
                    raise DatasetPathError(f"Missing required {path_type} directory for {year}: {path}")
        
        # Split Files
        train_files = [
            paths['voc07']['splits'] / "trainval.txt",
            paths['voc12']['splits'] / "trainval.txt"
        ]
        test_file = paths['voc07']['splits'] / "test.txt"

        # Validate split files
        for file in [*train_files, test_file]:
            if not file.exists():
                raise DatasetPathError(f"Missing required split file: {file}")
        
        # Process Split Files
        train_paths = []
        for train_file, year_path in zip(train_files, [paths['voc07'], paths['voc12']]):
            image_ids = [id.strip() for id in train_file.read_text().splitlines()]
            train_paths.extend([
                (year_path['images'] / f"{id}.jpg", year_path['annotations'] / f"{id}.xml") 
                for id in image_ids
            ])
        
        year_path = paths['voc07']
        image_ids = [id.strip() for id in test_file.read_text().splitlines()]
        test_paths = [
            (year_path['images'] / f"{id.strip()}.jpg", year_path['annotations'] / f"{id.strip()}.xml") 
            for id in image_ids
        ]

        # Image and Annotation file validation
        missing_files = [
            (img, annot) for img, annot in [*train_paths, *test_paths]
            if not img.exists() or not annot.exists()
        ]
        if missing_files:
            raise DatasetPathError(f"Missing {len(missing_files)} files, first few: {missing_files[:3]}")

        # Create data samples and append to list
        logging.info("Starting to create data samples from the dataset.")

        train_data = []
        for image_path, annotation_path in tqdm(train_paths, desc="Creating train data", ncols=500):
            data_sample = get_data_sample(image_path, annotation_path)
            train_data.append(data_sample)

        test_data = []
        for image_path, annotation_path in tqdm(test_paths, desc="Creating test data", ncols=500):
            data_sample = get_data_sample(image_path, annotation_path)
            test_data.append(data_sample)

        logging.info(f"Succesfully created data splits with {len(train_data)} training and {len(test_data)} testing images")

        return train_data, test_data
    
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(f"Invalid dataset root path: {dataset_root} - {str(e)}") from e
    except Exception as e:
        raise DatasetPathError(f"Error processing dataset: {str(e)}") from e

def load_data(dataset_root: Union[str, Path], save_data: Optional[bool] = False) -> Tuple[List[ImageData], List[ImageData]]:
    """
    Load the Pascal VOC dataset.

    Args:
        dataset_root (Union[str, Path]): The root directory of the dataset.
        save_data (Optional[bool]): If True, save the loaded data to disk. Defaults to False.

    Returns:
        Tuple[List[ImageData], List[ImageData]]: A tuple containing two lists of ImageData objects.
            The first list contains the training data, and the second list contains the validation data.
    """
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {dataset_root}")

    data_dir = dataset_root / "processed"
    train_file = data_dir / "training_data.pkl"
    test_file = data_dir / "test_data.pkl"

    if train_file.exists() and test_file.exists():
        logging.info("Loading data from preprocessed files...")

        with open(train_file, "rb") as f:
            train_data = pickle.load(f)
        with open(test_file, "rb") as f:
            test_data = pickle.load(f)

        logging.info("Data loaded.")

        return train_data, test_data
    else:
        logging.info("Processing raw data..")
        train_data, test_data = create_data_splits(dataset_root)

        if save_data:
            try:
                data_dir.mkdir(exist_ok=True)
            except OSError as e:
                logging.error(f"Failed to create directory {data_dir}: {e}")
                raise
            
            try:
                with open(train_file, "wb") as f:
                    pickle.dump(train_data, f)
                with open(test_file, "wb") as f:
                    pickle.dump(test_data, f)
                logging.info(f"Processed data saved to {str(data_dir)}")
            except IOError as e:
                logging.error(f"Error saving processed data: {e}")
                raise

        return train_data, test_data