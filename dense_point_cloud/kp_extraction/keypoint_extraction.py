from ultralytics import YOLO
import numpy as np

from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO
import cv2
import random

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model
source = cv2.imread('../images/image_l.png')
# Predict with the model


# Extract results
def get_keypoints(source):
    results = model(source=source, show=False, save = False)[0] 
    keypoints = np.array(results.keypoints.xy.cpu())
    return keypoints

def get_roi(source):
    results = model(source=source, show=False, save = False)[0] 
    roi = np.array(results.boxes.xyxy.cpu())
    return roi


def aplicar_mascara_imagen(image, mask, coordinates):
    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for coor in coordinates:
        mask[coor[1]:coor[3], coor[0]:coor[2]] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image

# Carga la imagen original y crea una máscara inicial

mask = np.zeros(source.shape[:2], dtype=np.uint8)  # Asegúrate de que la máscara sea del mismo tamaño que la imagen

roi = get_roi(source)
# Aplica la máscara
result_image = aplicar_mascara_imagen(source, mask, roi)

# Guarda o muestra la imagen resultante
cv2.imwrite('imagen_resultante.jpg', result_image)
