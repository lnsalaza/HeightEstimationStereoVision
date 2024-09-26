import cv2
import numpy as np
from .triangulation import find_depth_from_disparities

from .featuresExtractor import FaceFeatures, FeaturesExtractor

from time import sleep
from typing import Callable


def computeDepth(keypoinsL, keypoinsR, cams_sep, f_length):
    """Compute depth from keypoints

    Parameteres:
        keypointsL (np.ndarray): nx2 numpy array containing keypoints coordinates as viewed from left camera
        keypointsR (np.ndarray): nx2 numpy array containing keypoints coordinates as viewed from right camera
        cams_sep (float): horizontal distance between camera
        f_length (float): focal distance in px units calculated during calibration
    Returns:
        float: depth
    """
    return find_depth_from_disparities(keypoinsL[:, 0], keypoinsR[:, 0],
                                       cams_sep, f_length)


def computeHeigth(features: FaceFeatures, pixel_size: float):
    """Compute person heigth from body proportions
    Parameters:
        features (FaceFeatures): face features
        pixel_size: pixel_size at given depth in mm
    Returns:
        float: estimated height

    """
    mid_eye = np.mean((features.eye1, features.eye2), axis=0)
    mouth_eye = ((mid_eye[0]-features.mouth[0])**2 +
                 (mid_eye[1]-features.mouth[1])**2) ** 0.5
    head_size = mouth_eye * pixel_size * 3
    height = head_size * 8
    return height


def computeHeigth2(features: FaceFeatures, pixel_size: float, cam_center):
    """Compute person heigth realtive to the camera
    Parameters:
        features (FaceFeatures): face features
        pixel_size: pixel_size at given depth in mm
    Returns:
        float: estimated height

    """
    mid_eye_y = np.mean((features.eye1, features.eye2), axis=0)[1]
    cam_center_y = cam_center[1]

    height = (mid_eye_y-cam_center_y)*pixel_size * -1

    #  height = head_size * 8
    return height




def compute_height_using_face_metrics(img_left, img_right, baseline, fx, camera_center_left):
    """
    Calculates the height of an object using facial metrics from two camera images.

    Parameters:
    img_left (ndarray): Image from the left camera.
    img_right (ndarray): Image from the right camera.
    baseline (float): Distance between the two cameras (in meters).
    fx (float): Focal length of the cameras (in pixels).
    camera_center_left (tuple): Coordinates of the left camera's center (x, y).

    Returns:
    float: The calculated height of the object.

    Process:
    1. Extract keypoints from both images.
    2. Compute depth using extracted features.
    3. Calculate pixel size from depth and focal length.
    4. Determine height using left image features and pixel size.
    """
    
    features_left = FeaturesExtractor()
    features_right = FeaturesExtractor()

    # Extract keypoints from the images
    features_left = features_left.extract_keypts(img_left)
    features_right = features_right.extract_keypts(img_right)
    
    # Compute depth from the extracted features
    depth = computeDepth(features_left[2], features_right[2], baseline, fx)
    
    # Calculate pixel size based on depth and focal length
    px_size = depth / fx
    
    # Calculate height using features and pixel size
    height = computeHeigth2(features_left[1], px_size, camera_center_left)

    return height
