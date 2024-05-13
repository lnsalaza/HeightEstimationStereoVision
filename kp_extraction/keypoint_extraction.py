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
results = model(source=source, show=False, save = False)[0] 


# Extract results
keypoints = np.array(results.keypoints.xy.cpu())
roi = np.array(results.boxes.xyxy.cpu())

# mask = np.zeros(source.shape, float)

# x1 = mask[0][0]
# x2 = mask[0][2]

# y1 = mask[0][1]
# y2 = mask[0][3]


# for i in range(int(x1), int(x2)):
#     for j in range(int(y1), int(y2)):
#         print("X: " + str(i) + "\n Y: " + str(j))
#print(mask)

def aplicar_mascara_imagen(image, mask, coordinates):
    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for coor in coordinates:
        mask[coor[1]:coor[3], coor[0]:coor[2]] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image

# Carga la imagen original y crea una máscara inicial

mask = np.zeros(source.shape[:2], dtype=np.uint8)  # Asegúrate de que la máscara sea del mismo tamaño que la imagen

# Define tus coordenadas como una lista de tuplas (x1, y1, x2, y2)

# Aplica la máscara
result_image = aplicar_mascara_imagen(source, mask, roi)

# Guarda o muestra la imagen resultante
cv2.imwrite('imagen_resultante.jpg', result_image)
# cv2.imshow('Imagen con Máscara', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for person in keypoints:
#     red = random.randint(0,255)
#     green = random.randint(0,255)
#     blue = random.randint(0,255)
#     for kpt in person:
#         x, y = int(kpt[0]), int(kpt[1])

#         cv2.circle(source, (x, y), radius=3, color=(red, green, blue), thickness=-1)
    
# cv2.imshow("Keypoints without lines", source)
# cv2.waitKey(0)
# cv2.destroyAllWindows()