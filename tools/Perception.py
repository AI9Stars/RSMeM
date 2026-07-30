import argparse
import mimetypes
import re

from pathlib import Path
from fastmcp import FastMCP
import requests
import logging
import json

from utils import read_image, read_image_uint8
from service_urls import (
    BRIGHT_API_URL,
    CHANGEOS_API_URL,
    REMOTESAM_API_URL,
    REMOTECLIP_CLASSIFY_URL,
    REMOTECLIP_HEALTH_URL,
    SAM2_API_BASE,
    SM3DET_API_URL,
    STRIP_RCNN_DETECT_URL,
)

import torch
import numpy as np
from osgeo import gdal

import sys as _sys
import functools as _functools
# MCP stdio transport uses stdout as the JSONRPC channel; stray print() output
# on stdout corrupts message framing and deadlocks tool calls. Route all
# diagnostic prints in this module to stderr (does not affect tool return values).
print = _functools.partial(print, file=_sys.stderr)

mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def resolve_repo_path(path: str) -> str:
    """Resolve a client-supplied image path to an ABSOLUTE path the services can read.

    Deployed vision services may resolve the path we POST against their own
    working directory rather than this repository. The pipeline passes relative
    paths (``benchmark/data/qNNN/...``) whose real files live at the repo root, so
    a service may otherwise miss them. We therefore turn such relative paths into
    an absolute repo-root path here, before POSTing:

    - already-absolute path that exists -> returned unchanged;
    - relative path -> first existing match against ``[Path.cwd(), repo root]``
      (repo root = this file's ``parent.parent`` since ``Perception.py`` is in
      ``tools/``);
    - anything unresolvable -> returned unchanged (let the service report it).
    """
    if not path:
        return path
    candidate = Path(path)
    if candidate.is_absolute() and candidate.exists():
        return path
    for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
        resolved = (base / path).resolve()
        if resolved.exists():
            return str(resolved)
    return path


def _normalize_count_label(label: str) -> set[str]:
    """Normalize InstructSAM labels across spaces/underscores/camelCase/plurals."""
    raw_label = str(label)
    variants: set[str] = set()
    zh_synonyms = {
        "棒球": {"baseball field", "baseball diamond", "baseball", "baseballfield", "baseballdiamond"},
        "足球": {"soccer field", "football field", "soccer", "football"},
        "田径": {"track field", "athletic field", "sports field", "stadium"},
        "建筑": {"building"},
        "道路": {"road"},
        "停车": {"parking lot", "parking"},
        "树": {"tree"},
        "水": {"water body", "water"},
    }
    for marker, mapped in zh_synonyms.items():
        if marker in raw_label:
            variants.update(mapped)

    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", raw_label)
    spaced = re.sub(r"[^A-Za-z0-9]+", " ", spaced).lower().strip()
    words = [word[:-1] if word.endswith("s") and len(word) > 3 else word for word in spaced.split()]
    compact = "".join(words)
    normalized = " ".join(words)
    variants.update({normalized, compact})

    synonyms = {
        "baseball diamond": {"baseball field", "baseball diamond", "baseball"},
        "baseballdiamond": {"baseballfield", "baseballdiamond", "baseball"},
        "baseball field": {"baseball field", "baseball diamond", "baseball"},
        "baseballfield": {"baseballfield", "baseballdiamond", "baseball"},
        "airplane": {"airplane", "aircraft", "plane"},
        "aircraft": {"airplane", "aircraft", "plane"},
        "plane": {"airplane", "aircraft", "plane"},
    }
    for key, mapped in synonyms.items():
        if key in normalized or key in compact:
            variants.update(mapped)
    for variant in list(variants):
        variants.update(synonyms.get(variant, set()))
    return {variant for variant in variants if variant}

# def read_image(file_path: str) -> np.ndarray:
#     ds = gdal.Open(file_path)
#     if ds is None:
#         raise RuntimeError(f"Failed to open {file_path}")
    
#     bands = ds.RasterCount
#     if bands == 1:
#         img = ds.GetRasterBand(1).ReadAsArray()
#     else:
#         img = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
#         img = np.transpose(img, (1, 2, 0))

#     ds = None
#     return img
@mcp.tool(description="""
Perform threshold-based segmentation on a single-band raster image.

The function reads a raster image from the specified path, converts it to a binary mask
by applying a fixed threshold, and writes the resulting binary image to a new file.
Pixel values greater than the threshold are set to 255 (white), and values less than or
equal to the threshold are set to 0 (black).

Parameters:
    input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
    threshold (float or int): Pixel intensity threshold used to generate the binary mask.
    output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                        e.g., "question17/threshold_segmentation_2022-01-16.tif".

Returns:
    str: Message indicating the file path where the result is saved.
""")
def threshold_segmentation(input_image_path, threshold, output_path):
    '''
    Perform threshold-based segmentation on a single-band raster image.

    The function reads a raster image from the specified path, converts it to a binary mask
    by applying a fixed threshold, and writes the resulting binary image to a new file.
    Pixel values greater than the threshold are set to 255 (white), and values less than or
    equal to the threshold are set to 0 (black).

    Parameters:
        input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
        threshold (float or int): Pixel intensity threshold used to generate the binary mask.
        output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                           e.g., "question17/threshold_segmentation_2022-01-16.tif".

    Returns:
        str: Message indicating the file path where the result is saved.
    '''
    import os
    import rasterio
    import numpy as np

    with rasterio.open(input_image_path) as src:
        image = src.read(1)
        meta = src.meta.copy()

    binary_image = (image > threshold).astype(np.uint8) * 255

    meta.update(dtype=rasterio.uint8, count=1)
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **meta) as dst:
        dst.write(binary_image, 1)

    return f'Result save at {TEMP_DIR / output_path}'


@mcp.tool(description="""
Expands bounding boxes by a given radius and returns the expanded bounding boxes.

Parameters:
    bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
    radius (float): Expansion radius in the same unit as the GSD.
    gsd (float): Ground Sampling Distance in the same unit as the radius.

Returns:
    list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
""")
def bbox_expansion(bboxes: list[list[float]], radius: float, gsd: float):
    """
    Expands bounding boxes by a given radius and returns the expanded bounding boxes.

    Parameters:
        bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
        radius (float): Expansion radius in the same unit as the GSD.
        gsd (float): Ground Sampling Distance in the same unit as the radius.

    Returns:
        list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
    """
    expanded_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = x1 - radius / gsd
        y1 = y1 - radius / gsd
        x2 = x2 + radius / gsd
        y2 = y2 + radius / gsd
        expanded_bboxes.append([x1, y1, x2, y2])

    return expanded_bboxes



@mcp.tool(description="""
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """)
def count_above_threshold(file_path: str, threshold: float):
    """
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """
    import numpy as np
    import rasterio
    with rasterio.open(file_path) as src:
        x = src.read(1)
    x = np.asarray(x)
    # Count elements greater than threshold
    count = np.sum(x > threshold)
    
    return int(count)



@mcp.tool(description=
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_skeleton_contours("binary_mask.png")
        12
    """)
def count_skeleton_contours(image_path):
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_skeleton_contours("binary_mask.png")
        12
    """
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Binarize the image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Apply erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # Skeletonize
    skeleton = skeletonize(eroded > 0)  # Convert to boolean for skimage
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)



@mcp.tool(description=
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """)
def bboxes2centroids(bboxes):
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]



@mcp.tool(description=
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """)
def centroid_distance_extremes(centroids):
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """
    import numpy as np
    points = np.array(centroids)
    diff = points[:, None, :] - points[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    np.fill_diagonal(dist_matrix, np.inf)
    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_dist = dist_matrix[min_idx]

    np.fill_diagonal(dist_matrix, -np.inf)
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    max_dist = dist_matrix[max_idx]

    return {
        "min": (int(min_idx[0]), int(min_idx[1]), float(min_dist)),
        "max": (int(max_idx[0]), int(max_idx[1]), float(max_dist))
    }



@mcp.tool(description="""
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """)
def calculate_bbox_area(bboxes, gsd=None):
    """
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """
    total_area = 0.0
    for bbox in bboxes:
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected [x, y, w, h].")
        _, _, w, h = bbox
        area = w * h
        total_area += area

    if gsd is not None:
        total_area *= gsd * gsd
    
    return total_area
   
def get_model_output(model_name: str, input_image_path: str, **args):
    import pandas as pd

    results = pd.read_csv('../model_results.csv')
    result = None
    try:
        # classification
        if model_name in ['MSCN', 'RemoteCLIP']:
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
        # detection
        elif model_name == 'Strip-R-CNN':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
        # visual grounding
        elif model_name == 'RemoteSAM':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['text_prompt']]
        # counting
        elif model_name == 'InstructSAM':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['text_prompt']]
        # segmentation
        elif model_name == 'SAM2':
            result = results[(results['model'] == model_name) & (results['input_image_path'] == input_image_path)].values[0]
            result = result[args['bbox']]
    except:
        pass
    
    if result is None:
        return 'Failed to call model'
    else:
        return result



# @mcp.tool(description="""
# MSCN is a scene and land-use image classifier, effective for categories such as 
# Airport, BareLand, BaseballField, Beach, Bridge, Center, Church, Commercial, 
# DenseResidential, Desert, Farmland, Forest, Industrial, Meadow, MediumResidential, 
# Mountain, Park, Parking, Playground, Pond, Port, RailwayStation, Resort, River, 
# School, SparseResidential, Square, Stadium, StorageTanks, and Viaduct.

# Parameters:
# - input_image_path (str): Path to the input image.

# Returns:
# - np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]

# Example:
# array([
#   'MSCN',
#   'benchmark/data/question189/J.jpg',
#   'Resort',
#   0.7052103281021118,
#   [
#     ('Resort', 0.7052103281021118),
#     ('StorageTanks', 0.11459718644618988),
#     ('Desert', 0.019159140065312386),
#     ('Meadow', 0.013844668865203857),
#     ('Beach', 0.013844599016010761)
#   ]
# ], dtype=object)
# """)
# def MSCN(input_image_path):
#     return get_model_output('MSCN', input_image_path)


@mcp.tool(description="""
RemoteCLIP is a scene and land-use image classifier, specialized for categories such as 
Airport, Beach, Bridge, Commercial, Desert, Farmland, FootballField, Forest, Industrial, 
Meadow, Mountain, Park, Parking, Pond, Port, RailwayStation, Residential, River, and Viaduct.

Parameters:
- input_image_path (str): Path to the input image.

Returns:
- np.ndarray: [model_name, image_path, predicted_class, confidence, top-5 predictions]

Example:
array([
  'RemoteCLIP',
  'benchmark/data/question189/J.jpg',
  'Resort',
  0.7052103281021118,
  [
    ('Resort', 0.7052103281021118),
    ('StorageTanks', 0.11459718644618988),
    ('Desert', 0.019159140065312386),
    ('Meadow', 0.013844668865203857),
    ('Beach', 0.013844599016010761)
  ]
], dtype=object)
""")
def RemoteCLIP(input_image_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("MCP_RemoteCLIP_Tool")

    # Resolve the RemoteCLIP inference and health-check endpoints.
    REMOTECLIP_API_URL = REMOTECLIP_CLASSIFY_URL
    HEALTH_CHECK_URL = REMOTECLIP_HEALTH_URL
    try:
        health_response = requests.get(HEALTH_CHECK_URL, timeout=10)
        health_response.raise_for_status()
        if health_response.json()["status"] != "ready":
            raise RuntimeError("RemoteCLIP API service not ready")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"RemoteCLIP API Health checkFailed: {str(e)}")

    # Validate the input image before uploading it.
    input_image_path = resolve_repo_path(input_image_path)
    input_image = Path(input_image_path)
    if not input_image.exists():
        raise FileNotFoundError(f"Input imagedoes not exist: {input_image_path}")

    # Submit the image to the classification API.
    try:
        with open(input_image, "rb") as f:
            files = {"file": (input_image.name, f, "image/jpeg")}
            logger.info(f"Calling RemoteCLIP API to classify image: {input_image.name}")
            response = requests.post(REMOTECLIP_API_URL, files=files, timeout=300)
        
        # Surface service-side errors with their response details.
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            raise RuntimeError(f"API callFailed: {error_detail}")
        
        result = response.json()
        logger.info(f"Classification completed, predicted class: {result['predicted_class']}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API Communication failed: {str(e)}")

    # Preserve the historical NumPy-array return schema used by the agent tools.
    return np.array([
        "RemoteCLIP",  # model_name
        input_image_path,  # image_path
        result["predicted_class"],  # predicted_class
        result["confidence"],  # confidence
        result["top5_predictions"]  # top-5 predictions
    ], dtype=object)
    # return get_model_output('RemoteCLIP', input_image_path)



@mcp.tool(description="""
The Strip R-CNN is a general-purpose oriented object detector specifically 
engineered for complex aerial and satellite imagery.
Its core Strip Pooling mechanism efficiently addresses the large scale and
extreme aspect ratio variations common in remote sensing. Compared to SM3Det, it is particularly
specialized in detecting and localizing different types of ships and naval vessels.

This model is highly effective at detecting the following categories:
Aircraft: PassengerAircraft, FighterAircraft, Helicopter, TransportAircraft,
          TrainerAircraft, OthersAircraft, ExecutiveAircraft, UAV, CargoAircraft, BomberAircraft
Vessel: EngineeringShip, FishingVessel, Motorboat, Tugboat, SailingShip,
           WarShip, Destroyer, CargoShip, AircraftCarrier, OthersVessel
Vehicle: Bus, Truck, EngineeringVehicle, Trailer, Motorhome, Excavator,
           DumpTruck, Tank, OthersVehicle, Tractor
Port Facilities: Crane, ContainerStack, OilTank, LightTower, 
          Bridge, OthersFacility, BaseballField

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the ship type to detect.

Returns:
- list[list[float]]: A list of bounding boxes, each represented as 
  [x_min, y_min, x_max, y_max].

Example:
Input:
  input_image_path = "benchmark/data/questionXXX/ship_example.png"
  text_prompt = "L3 aircraft carrier"

Output:
  [
    [120.5, 340.7, 480.2, 600.9],
    [700.3, 220.1, 950.6, 400.4]
  ]
""")
def Strip_R_CNN(input_image_path, text_prompt):
    API_URL = STRIP_RCNN_DETECT_URL
    input_image_path = resolve_repo_path(input_image_path)
    payload: Dict[str, str] = {
        "input_image_path": input_image_path,
        "text_prompt": text_prompt 
    }

    try:
        # Send the image path and text prompt to the detector.
        response = requests.post(API_URL, json=payload, timeout=30)
        
        # Convert HTTP 4xx/5xx responses into request exceptions.
        response.raise_for_status()
        
        # Parse the expected list of bounding boxes.
        try:
            bboxes: List[List[float]] = response.json()
            return bboxes
        
        except requests.exceptions.JSONDecodeError as json_err:
            print("JSON parsing error: API response is not valid JSON.")
            print(f"   Return status code: {response.status_code}")
            print(f"   Return content preview: {response.text[:200]}...")
            return []

    except requests.exceptions.RequestException as req_err:
        # Covers connection, timeout, DNS, and HTTP-status failures.
        print(f"❌ Request failed or HTTP error: {req_err}")
        return []
    
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return []
    # return get_model_output('Strip-R-CNN', input_image_path, text_prompt=text_prompt)

import os
from typing import List

@mcp.tool(description="""
SM3Det is a remote sensing object detection model. 
Given an input image and a natural language prompt specifying the target object 
(e.g., "plane", "ship", "storage tank"), it detects all instances of that object 
and returns their bounding boxes.

This model is particularly strong at detecting and localizing the following categories:
- plane
- ship
- storage tank
- baseball diamond
- tennis court
- basketball court
- ground track field
- harbor
- bridge
- large vehicle
- small vehicles
- helicopter
- roundabout
- soccer ball field
- swimming pool

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the object to detect.

Returns:
- list[list[float]]: A list of bounding boxes, each represented as 
  [x_min, y_min, x_max, y_max].
""")
def SM3Det(input_image_path: str, text_prompt: str) -> List[List[float]]:
    """
     FastAPI ，。
    """
    input_image_path = resolve_repo_path(input_image_path)
    payload = {
        "input_image_path": input_image_path,
        "text_prompt": text_prompt,
    }

    try:
        response = requests.post(SM3DET_API_URL, json=payload, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"SM3Det HTTP request failed: {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"SM3Det response is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise RuntimeError("SM3Det response format unexpected (expected list of bboxes).")

    return data


# @mcp.tool(description="""
# SM3Det is a remote sensing object detection model. 
# Given an input image and a natural language prompt specifying the target object 
# (e.g., "plane", "ship", "storage tank"), it detects all instances of that object 
# and returns their bounding boxes.

# This model is particularly strong at detecting and localizing the following categories:
# - plane
# - ship
# - storage tank
# - baseball diamond
# - tennis court
# - basketball court
# - ground track field
# - harbor
# - bridge
# - large vehicle
# - small vehicles
# - helicopter
# - roundabout
# - soccer ball field
# - swimming pool

# Parameters:
# - input_image_path (str): Path to the input image.
# - text_prompt (str): Natural language description of the object to detect.

# Returns:
# - list[list[float]]: A list of bounding boxes, each represented as 
#   [x_min, y_min, x_max, y_max].

# Example:
# Input:
#   input_image_path = "benchmark/data/question235/P0173.png"
#   text_prompt = "plane"

# Output:
#   [
#     [491.08, 532.47, 562.03, 598.47],
#     [548.89, 563.04, 636.54, 643.85],
#     [57.80, 335.57, 191.65, 446.29],
#     [401.37, 474.06, 509.87, 573.09],
#     [344.69, 146.72, 464.14, 249.53],
#     [736.10, 503.04, 809.28, 568.13],
#     [680.84, 448.89, 760.03, 512.00],
#     [588.72, 312.11, 666.23, 378.02],
#     [537.49, 258.38, 610.34, 313.70]
#   ]
# """)
# def SM3Det(input_image_path, text_prompt):
#     return get_model_output('SM3Det', input_image_path, text_prompt=text_prompt)

@mcp.tool(description="""
RemoteSAM is a remote sensing visual grounding model. Given an input image and a text prompt 
describing a region of interest (e.g., "the football field located on the westernmost side"), 
it outputs the corresponding bounding box coordinates.

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the target object/region.

Returns:
- list[int]: Bounding box [x_min, y_min, x_max, y_max]

Example:
Input:
  input_image_path = "benchmark/data/question226/478549_4934011_2048_32610_sport_soccer.jpg"
  text_prompt = "the football field located on the westernmost side"

Output:
  [0, 264, 127, 342]
""")
def RemoteSAM(input_image_path, text_prompt):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MCP_RemoteSAM_Tool")
    """Call RemoteSAM and return ``[x_min, y_min, x_max, y_max]``."""

    API_ENDPOINT = REMOTESAM_API_URL
    
    input_image_path = resolve_repo_path(input_image_path)
    payload = {
        "input_image_path": input_image_path,
        "text_prompt": text_prompt
    }
    
    try:
        # Submit one prompt-grounded detection request.
        response = requests.post(
            API_ENDPOINT, 
            json=payload, 
            timeout=300,
        )
        
        response.raise_for_status()
        
        result = response.json()
        
        # The service returns a four-element integer list.
        if isinstance(result, list) and all(isinstance(x, int) for x in result):
            return result
        else:
            raise ValueError(f"API returned an invalid list structure: {result}")

    except requests.exceptions.HTTPError as e:
        # Preserve the detail returned by FastAPI for HTTP failures.
        try:
            error_detail = response.json().get('detail', 'No detailed error message.')
        except:
            error_detail = f"HTTP Error {response.status_code}"
            
        print(f"RemoteSAM API HTTP Error at {API_ENDPOINT}: {error_detail}")
        raise RuntimeError(f"API communication failed: {error_detail}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling RemoteSAM API at {API_ENDPOINT}: {e}")
        raise RuntimeError(f"Failed to communicate with RemoteSAM API: {e}")
    except ValueError as e:
        print(f"RemoteSAM API data parsing issue: {e}")
        raise ValueError(f"RemoteSAM API execution failed: {e}")
   
    # return get_model_output('RemoteSAM', input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""
InstructSAM is an instruction-guided counting model for remote sensing images. 
Given an input image and a natural language prompt specifying the target object 
(e.g., "storage tank", "football field"), it detects and counts the number of 
instances matching the description.

Parameters:
- input_image_path (str): Path to the input image.
- text_prompt (str): Natural language description of the object to count.

Returns:
- int: The number of objects in the image that match the text prompt.

Example:
Input:
  input_image_path = "benchmark/data/question231/B.jpg"
  text_prompt = "storage tank"

Output:
  8
""")
def InstructSAM(input_image_path, text_prompt):
    # The endpoint is configurable for local or remote deployment.
    FASTAPI_SERVICE_URL = os.getenv(
        'INSTRUCT_SAM_COUNTING_ENDPOINT', 
        "http://127.0.0.1:5559/count"
    )

    input_image_path = resolve_repo_path(input_image_path)
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image file not found at: {input_image_path}")

    content_type = mimetypes.guess_type(input_image_path)[0] or "image/jpeg"
    image_handle = open(input_image_path, 'rb')
    files = {
        'image_file': (Path(input_image_path).name, image_handle, content_type)
    }
    
    data = {
        'visualize': 'False'
    }

    try:
        # Upload the image to the counting endpoint.
        response = requests.post(FASTAPI_SERVICE_URL, files=files, data=data, timeout=600)
        response.raise_for_status()
        
        result: Dict[str, Any] = response.json()
        count_data: Dict[str, int] = result.get('count_data', {})
        
        if not count_data:
            return 0

        target_variants = _normalize_count_label(text_prompt)
        total_count = 0
        
        for category, count in count_data.items():
            category_variants = _normalize_count_label(category)
            if target_variants & category_variants:
                 total_count += count
                 
        return total_count if total_count > 0 else 0
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling InstructSAM FastAPI service at {FASTAPI_SERVICE_URL}: {e}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 0
    finally:
        image_handle.close()
    # return get_model_output('InstructSAM', input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""
    Use SAM2 to segment the input image and return the bounding box.

    Parameters:
        input_image_path (str): Path to the input image.
        bbox (list): Bounding box of the segmented object.

    Returns:
        str: Path to the segmented image.
""")
def SAM2(input_image_path, bbox, output_path):
    # Configure the SAM2 client and validate service readiness.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("MCP_SAM2_Tool")

    # SAM2 endpoints served by the configured FastAPI instance.
    SEGMENT_ENDPOINT = f"{SAM2_API_BASE}/segment"
    HEALTH_ENDPOINT = f"{SAM2_API_BASE}/health"

    SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "bmp", "tiff"}
    try:
        health_response = requests.get(HEALTH_ENDPOINT, timeout=10)
        health_response.raise_for_status()
        health_status = health_response.json()
        if health_status["status"] != "ready":
            raise RuntimeError(f"SAM2 service not ready: {health_status}")
        logger.info(f"SAM2 service is healthy (device: {health_status['device']})")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to check SAM2 service health: {str(e)}")

    # Validate the input file and supported image extension.
    input_image_path = resolve_repo_path(input_image_path)
    input_image = Path(input_image_path)
    if not input_image.exists():
        raise FileNotFoundError(f"Input image not found: {input_image_path}")
    
    image_ext = input_image.suffix.lower().lstrip(".")
    if image_ext not in SUPPORTED_IMAGE_FORMATS:
        raise ValueError(
            f"Unsupported image format: {image_ext}. Supported formats: {SUPPORTED_IMAGE_FORMATS}"
        )

    # Encode the optional bounding box as the API's comma-separated parameter.
    bbox_str = None
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"BBox must have 4 elements (x, y, w, h), got {len(bbox)}")
        try:
            bbox_int = tuple(map(int, bbox))
        except ValueError:
            raise TypeError("BBox elements must be integers")
        bbox_str = ",".join(map(str, bbox_int))

    # Submit the segmentation request.
    try:
        with open(input_image, "rb") as f:
            files = {"image": (input_image.name, f, f"image/{image_ext}")}
            data = {"bbox_str": bbox_str} if bbox_str else {}

            logger.info(
                f"Calling SAM2 segment API: image={input_image.name}, bbox={bbox_str or 'none'}"
            )
            response = requests.post(
                SEGMENT_ENDPOINT,
                files=files,
                data=data,
                timeout=300,
            )

        # Convert known API failures into actionable exceptions.
        if response.status_code == 400:
            error_detail = response.json().get("detail", "Unknown error")
            raise ValueError(f"SAM2 API invalid request: {error_detail}")
        elif response.status_code == 503:
            error_detail = response.json().get("detail", "Unknown error")
            raise RuntimeError(f"SAM2 service unavailable: {error_detail}")
        elif response.status_code == 500:
            error_detail = response.json().get("detail", "Unknown error")
            raise RuntimeError(f"SAM2 inference failed: {error_detail}")
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"SAM2 API request failed: {str(e)}")

    # Validate the response schema before returning bounding boxes.
    try:
        api_result = response.json()
        bounding_boxes = api_result["bounding_boxes"]
        if not isinstance(bounding_boxes, list):
            raise TypeError("API returned invalid bounding boxes (expected list)")
        for bbox in bounding_boxes:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                raise ValueError(f"Invalid bbox format: {bbox} (expected [x, y, w, h])")
        return bounding_boxes

    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Failed to parse SAM2 API result: {str(e)}")
    # return get_model_output('SAM2', input_image_path, bbox=bbox, output_path=output_path)
try:
    import changeos
except ImportError:
    changeos = None
import numpy as np
import cv2
import uuid
from pathlib import Path
import logging
from datetime import datetime

def read_image(image_path: str,device) -> torch.Tensor:
    """Load an image, normalize it, and move the tensor to the target device."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable toRead image: {image_path}")
    # OpenCV loads BGR; the model expects RGB.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to the model input resolution.
    img_resized = cv2.resize(img_rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # [C, H, W]
    # Normalize 8-bit pixels to [0, 1].
    img_tensor = img_tensor / 255.0
    return img_tensor.to(device)


def generate_output_path(pre_path: Path, post_path: Path, is_building: bool) -> Path:
    """Create a unique output path for building extraction or change detection."""
    default_output_dir = Path("./changeos_results")
    default_output_dir.mkdir(parents=True, exist_ok=True)
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    filename = f"building_extract_{pre_path.stem}_{unique_id}.png" if is_building else f"change_detect_{pre_path.stem}_vs_{post_path.stem}_{unique_id}.png"
    return default_output_dir / filename

import logging
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

import logging
import requests
from pathlib import Path
from datetime import datetime

@mcp.tool(description="""
ChangeOS is a change detection and building extraction model for remote sensing images.
It can detect changes between two images or extract buildings from a single image.

Parameters:
- pre_image_path (str): Path to the pre-event image.
- post_image_path (str): Path to the post-event image. 
  (Use the same path as pre_image_path for building extraction)
- output_path (str): Path to save the output visualization.

Returns:
- str: Path to the saved output image.
""")
def ChangeOS(pre_image_path: str, post_image_path: str, output_path: str) -> str:
    """
    APIChangeOS
    
    Args:
        pre_image_path: 
        post_image_path: ()
        output_path: 
        
    Returns:
        
        
    Raises:
        FileNotFoundError: 
        RuntimeError: API
    """
    
    # ==========  ==========
    log_dir = Path("logs/changeos")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"changeos_{timestamp}.log"
    
    logger = logging.getLogger(f"MCP_ChangeOS_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ========== API  ==========
    API_ENDPOINT = CHANGEOS_API_URL
    
    logger.info("=" * 80)
    logger.info("ChangeOS MCP Tool Execution Started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"API Endpoint: {API_ENDPOINT}")
    logger.info(f"Pre-image: {pre_image_path}")
    logger.info(f"Post-image: {post_image_path}")
    logger.info(f"Output path: {output_path}")
    
    start_time = datetime.now()
    
    try:
        # 1. Resolve inputs to absolute repo-root paths via the shared client
        #    helper so the service can read them regardless of its CWD. This
        #    replaces the previous buggy repo base (.parent.parent.parent.parent),
        #    which only worked when CWD happened to be the repo root.
        logger.debug(f"Current working directory: {Path.cwd()}")
        pre_image_path = resolve_repo_path(pre_image_path)
        post_image_path = resolve_repo_path(post_image_path)
        logger.info(f"Resolved pre-image path: {pre_image_path}")
        logger.info(f"Resolved post-image path: {post_image_path}")
        
        # 2. API
        payload = {
            "pre_image_path": pre_image_path,
            "post_image_path": post_image_path,
            "output_path": output_path
        }
        
        logger.info("Sending request to ChangeOS API...")
        logger.debug(f"Payload: {payload}")
        
        # 3. POST
        try:
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                timeout=300  # 5
            )
            
            # HTTP
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 300 seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to API at {API_ENDPOINT}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(f"API connection failed. Is the service running? {error_msg}")
            
        except requests.exceptions.HTTPError as e:
            # Preserve the service-provided error detail when available.
            try:
                error_detail = response.json().get('detail', 'No detailed error message.')
            except:
                error_detail = f"HTTP Error {response.status_code}"
            
            error_msg = f"API returned error: {error_detail}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Parse and validate the ChangeOS response.
        try:
            result = response.json()
            logger.debug(f"API Response: {result}")
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"API execution failed: {error_msg}")
                raise RuntimeError(f"ChangeOS API execution failed: {error_msg}")
            
            output_path_result = result.get('output_path')
            task_type = result.get('task_type')
            inference_time = result.get('inference_time')
            total_time = result.get('total_time')
            
            logger.info(f"Task type: {task_type}")
            logger.info(f"Inference time: {inference_time:.2f}s")
            logger.info(f"API total time: {total_time:.2f}s")
            logger.info(f"Output saved to: {output_path_result}")
            
            # Verify that the service-reported output file was created.
            output_file = Path(output_path_result)
            if output_file.exists():
                file_size = output_file.stat().st_size / 1024  # KB
                logger.info(f"Output file size: {file_size:.2f} KB")
            else:
                logger.warning(f"Output file not found at: {output_path_result}")
            
            client_total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Client total time: {client_total_time:.2f}s")
            logger.info("ChangeOS MCP Tool Execution Completed Successfully")
            logger.info("=" * 80)
            
            return output_path_result
            
        except (ValueError, KeyError) as e:
            error_msg = f"Failed to parse API response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
    except Exception as e:
        logger.error(f"ChangeOS execution failed: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise
        
    finally:
        # handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

import requests
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

@mcp.tool(description="""
BRIGHT (SiamCRNN) is a model for Building Damage Assessment in remote sensing images.
It compares pre-event and post-event images to generate a damage classification map 
(e.g., No Damage, Minor, Major, Destroyed).  
--- Damage Classes and Colors (RGB) ---
- **0: No damage** (255, 255, 255) -> White
- **1: Minor damage** (70, 181, 121) -> Green
- **2: Major damage** (228, 189, 139) -> Yellow/Tan
- **3: Destroyed** (182, 70, 69) -> Red
Parameters:
- pre_image_path (str): Path to the pre-event image (must be provided).
- post_image_path (str): Path to the post-event image (must be provided).
- output_path (str): Path to save the output damage classification visualization (e.g., /path/to/result.png).

Returns:
- str: Path to the saved output colored damage map image.
""")
def BRIGHT_DamageAssessment(pre_image_path: str, post_image_path: str, output_path: str) -> str:
    """
    APIBRIGHT/SiamCRNN。
    
    Args:
        pre_image_path: 
        post_image_path: 
        output_path: 
        
    Returns:
        
        
    Raises:
        FileNotFoundError: 
        RuntimeError: API
    """
    
    # ==========  ==========
    log_dir = Path("logs/bright_damage_assessment")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bright_{timestamp}.log"
    
    logger = logging.getLogger(f"MCP_BRIGHT_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ========== API  (8120) ==========
    API_ENDPOINT = BRIGHT_API_URL
    
    logger.info("=" * 80)
    logger.info("BRIGHT Damage Assessment MCP Tool Execution Started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"API Endpoint: {API_ENDPOINT}")
    logger.info(f"Pre-image: {pre_image_path}")
    logger.info(f"Post-image: {post_image_path}")
    logger.info(f"Output path: {output_path}")
    
    start_time = datetime.now()
    
    try:
        # 1. Resolve inputs to absolute repo-root paths so the service can read
        #    them regardless of its CWD (Option B client-side data-path fix).
        #    output_path is left as-is (the service creates it).
        pre_image_path = resolve_repo_path(pre_image_path)
        post_image_path = resolve_repo_path(post_image_path)
        
        # 2. API
        payload = {
            "pre_image_path": pre_image_path,
            "post_image_path": post_image_path,
            "output_path": output_path
        }
        
        logger.info("Sending request to BRIGHT Damage Assessment API...")
        logger.debug(f"Payload: {payload}")
        
        # 3. POST
        try:
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                timeout=300  # 5
            )
            
            # HTTP
            response.raise_for_status()
            
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after 300 seconds"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Failed to connect to API at {API_ENDPOINT}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(f"API connection failed. Is the service running on port 8120? {error_msg}")
            
        except requests.exceptions.HTTPError as e:
            # Preserve the service-provided error detail when available.
            try:
                error_detail = response.json().get('detail', 'No detailed error message.')
            except:
                error_detail = f"HTTP Error {response.status_code}"
            
            error_msg = f"API returned error: {error_detail}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Parse and validate the BRIGHT response.
        try:
            result: Dict[str, Any] = response.json()
            logger.debug(f"API Response: {result}")
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                logger.error(f"API execution failed: {error_msg}")
                raise RuntimeError(f"BRIGHT API execution failed: {error_msg}")
            
            output_path_result = result.get('output_path')
            task_type = result.get('task_type')
            inference_time = result.get('inference_time')
            total_time = result.get('total_time')
            
            logger.info(f"Task type: {task_type}")
            logger.info(f"Inference time: {inference_time:.2f}s")
            logger.info(f"API total time: {total_time:.2f}s")
            logger.info(f"Output saved to: {output_path_result}")
            
            # Verify that the service-reported output file was created.
            output_file = Path(output_path_result)
            if output_file.exists():
                file_size = output_file.stat().st_size / 1024  # KB
                logger.info(f"Output file size: {file_size:.2f} KB")
            else:
                logger.warning(f"Output file not found at: {output_path_result}")
            
            client_total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Client total time: {client_total_time:.2f}s")
            logger.info("BRIGHT Damage Assessment MCP Tool Execution Completed Successfully")
            logger.info("=" * 80)
            
            return output_path_result
            
        except (ValueError, KeyError) as e:
            error_msg = f"Failed to parse API response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
        
    except Exception as e:
        logger.error(f"BRIGHT execution failed: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise
        
    finally:
        # Avoid accumulating duplicate handlers across repeated MCP calls.
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    mcp.run()
