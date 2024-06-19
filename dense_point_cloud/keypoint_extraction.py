import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------- KEYPOINTS EXTRACTION -------------------------------------------------------

# Load a model
model = YOLO('yolov8n-pose.pt').to(device=device)  # load an official model



# Extract results
def get_keypoints(source):
    results = model(source=source, show=False, save = False, conf=0.85)[0] 
    keypoints = np.array(results.keypoints.xy.cpu())
    return keypoints

def get_roi(source):
    results = model(source=source, show=False, save = False, conf=0.85)[0] 
    roi = np.array(results.boxes.xyxy.cpu())
    return roi

def apply_roi_mask(image, roi):
    mask = np.zeros(image.shape[:2], dtype=np.uint8) 

    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for coor in roi:
        mask[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image

def apply_keypoints_mask(image, keypoints):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for person in keypoints:
        for kp in person:
            y, x = int(kp[1]), int(kp[0])
            # Verificar si las coordenadas están dentro de los límites de la imagen
            if 0 <= y - 1 < image.shape[0] and 0 <= x - 1 < image.shape[1]:
                mask[y - 1, x - 1] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos
    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)
    return masked_image