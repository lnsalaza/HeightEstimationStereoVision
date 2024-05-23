import os
import cv2
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator


# Aplicar el filtro bilateral
sigma = 1.5  # Parámetro de sigma utilizado para el filtrado WLS.
lmbda = 8000.0  # Parámetro lambda usado en el filtrado WLS.


# NEW
LEFT_VIDEO = '../videos/rectified/distance_left.avi'
RIGHT_VIDEO = '../videos/rectified/distance_right.avi'

MATRIX_Q = '../config_files/newStereoMap.xml'
fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
Q = fs.getNode("disparity2depth_matrix").mat()
fs.release() 


#OLD 

# LEFT_VIDEO = "../videos/rectified/distance_left_calibrated.avi"
# RIGHT_VIDEO = "../videos/rectified/distance_right_calibrated.avi"

# MATRIX_Q = '../config_files/old_config/stereoMap.xml'
# fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
# Q = fs.getNode("disparityToDepthMap").mat()
# fs.release() 

# --------------------------------------------------- KEYPOINTS EXTRACTION -------------------------------------------------------

# Load a model
model = YOLO('yolov8n-pose.pt')  # load an official model



# Extract results
def get_keypoints(source):
    results = model(source=source, show=False, save = False, conf=0.85)[0] 
    keypoints = np.array(results.keypoints.xy.cpu())
    return keypoints

def get_roi(source):
    results = model(source=source, show=False, save = False, conf=0.85)[0] 
    roi = np.array(results.boxes.xyxy.cpu())
    return roi

def applyROImask(image, roi):
    mask = np.zeros(image.shape[:2], dtype=np.uint8) 

    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for coor in roi:
        mask[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image

def applyKeypointsMask(image, keypoints):
    mask = np.zeros(image.shape[:2], dtype=np.uint8) 
    centers = []

    # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
    for person in keypoints:
        center_x = 0
        center_y = 0
        total = 0

        for kp in person:
            center_x += kp[1]
            center_y += kp[0]
            total += 1
            mask[int(kp[1]),int(kp[0])] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos
        
        center_x = center_x/total
        center_y = center_y/total

        mask[int(center_x),int(center_y)] = 1
        centers.append([int(center_x),int(center_y)])

    print(centers)
    # Aplica la máscara a la imagen
    masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

    return masked_image


def save_image(path, image, image_name, grayscale=False):
    # Asegúrate de que el directorio existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Listar todos los archivos en el directorio
    files = os.listdir(path)

    # Filtrar los archivos que son imágenes (puedes ajustar los tipos según tus necesidades)
    image_files = [f for f in files if f.startswith(image_name)]

    # Determinar el siguiente número para la nueva imagen
    next_number = len(image_files) + 1

    # Crear el nombre del archivo para la nueva imagen
    new_image_filename = f'{image_name}_{next_number}.jpg'
    # Ruta completa del archivo
    full_path = os.path.join(path, new_image_filename)

    # Convertir a escala de grises si es necesario
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen usando cv2.imwrite
    cv2.imwrite(full_path, image)



# --------------------------------------------------- DENSE POINT CLOUD ----------------------------------------------------------

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        retval, frame = cap.read()
        if not retval:
            break
    
        frames.append(frame)
    
    cap.release()
    return frames


# EXTRACCION DE FRAME UNICO
def extract_image_frame(n_frame, color=True, save = True):
    frames_l = extract_frames(LEFT_VIDEO)
    frames_r = extract_frames(RIGHT_VIDEO)

    image_l = frames_l[n_frame]
    image_r = frames_r[n_frame]

    if not color:
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    if save:
        cv2.imwrite("../images/image_l.png", image_l)
        cv2.imwrite("../images/image_r.png", image_r)
    return image_l, image_r


def compute_disparity(left_image, right_image):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    blockSize_var = 7
    P1 = 8 * 3 * (blockSize_var ** 2)  
    P2 = 32 * 3 * (blockSize_var ** 2) 

    stereo = cv2.StereoSGBM_create(
        numDisparities = 68,
        blockSize = blockSize_var, 
        minDisparity=5,
        P1=P1,
        P2=P2,
        disp12MaxDiff=33,
        uniquenessRatio=10,
        preFilterCap=33,
        mode=cv2.StereoSGBM_MODE_HH
    )

    # stereo.setPreFilterType(0)
    # stereo.setPreFilterSize(7*2+5)
    # stereo.setPreFilterCap(63)
    # stereo.setTextureThreshold(10)
    # stereo.setUniquenessRatio(13)
    # stereo.setSpeckleRange(2)
    # stereo.setSpeckleWindowSize(3*2)
    # stereo.setDisp12MaxDiff(5)
    # stereo.setMinDisparity(5)

    # Calcular el mapa de disparidad de la imagen izquierda a la derecha
    left_disp = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

    # Crear el matcher derecho basado en el matcher izquierdo para consistencia
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Calcular el mapa de disparidad de la imagen derecha a la izquierda
    right_disp = right_matcher.compute(right_image, left_image).astype(np.float32) / 16.0

    # Crear el filtro WLS y configurarlo
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Filtrar el mapa de disparidad utilizando el filtro WLS
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)

    # Normalización para la visualización o procesamiento posterior
    filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_disp = np.uint8(filtered_disp)
    return filtered_disp

def disparity_to_pointcloud(disparity, Q, image, custom_mask=None):
    points_3D = cv2.reprojectImageTo3D(disparity, Q) 
    #colors = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    mask = disparity > 0

    if custom_mask is not None:
        mask  = custom_mask > 0

    out_points = points_3D[mask]
    #out_colors = colors[mask]
    out_colors = image[mask]

    return out_points, out_colors


# CREACIÒN DE NUBE DE PUNTOS
# 250 Y 500
# img_l, img_r = extract_image_frame(6900, False, False)

# 450 Y 600
# img_l, img_r = extract_image_frame(4710, False, False)

# 150 Y 500
# img_l, img_r = extract_image_frame(3930, False, False)

# 300
#img_l, img_r = extract_image_frame(700, True, False)

img_l = cv2.imread("../images/calibration_results/image_l.png")
img_r = cv2.imread("../images/calibration_results/image_r.png")

img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

disparity = compute_disparity(img_l, img_r)

baseline, fpx = 0, 0

with open("../config_files/stereoParameters.json", "r") as file:
    params = json.load(file)

    baseline = -(params["stereoT"][0])
    fpx = params["flCamera1"][0]

# 250 Y 500
# print(baseline * fpx / disparity[527][1075]) #Elihan
# print(baseline * fpx / disparity[471][730]) #Loberlly

# 450 Y 600
# print(baseline * fpx / disparity[510][890]) #Elihan
# print(baseline * fpx / disparity[530][1060]) #Loberlly

# 150 Y 500
# print(baseline * fpx / disparity[315][700]) #Elihan
# print(baseline * fpx / disparity[525][1055]) #Loberlly

# # 300
# print(baseline * fpx / disparity[490][830]) #Elihan
# print(baseline * fpx / disparity[480][1180]) #Loberlly

# 250 Y 500
# print("Depth Elihan: ", str(baseline * fpx / disparity[598][1110])) #Elihan
# print("Depth Loberlly: ", str(baseline * fpx / disparity[567][743])) #Loberlly


# Filtra los puntos deseados


# # ROI
# roi = get_roi(img_l)
# result_image = applyROImask(disparity, roi)
# save_image("../images/prediction_results/", result_image, "filtered_roi", False)
# eps, min_samples = 10, 100

# KEYPOINTS
keypoints = get_keypoints(img_l)
result_image = applyKeypointsMask(disparity, keypoints)
save_image("../images/prediction_results/", result_image, "filtered_keypoints", False)
eps, min_samples = 50, 6

# Obtener nube de puntos filtrada
point_cloud, colors = disparity_to_pointcloud(disparity, Q, img_l, result_image)
point_cloud = point_cloud.astype(np.float64)

# OBTENCION DE CENTROIDES Y VISUALIZACION

# Aplicar DBSCAN para encontrar clusters
db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
labels = db.labels_

# Obtener el número de clusters (excluyendo el ruido si lo hay)
unique_labels = set(labels)
if -1 in unique_labels:
    unique_labels.remove(-1)  # Remover la etiqueta de ruido si está presente

if not unique_labels:
    print("No hay clusters.")
else:
    # Crear una lista para almacenar los centroides
    centroids = []

    # Procesar cada cluster
    for label in unique_labels:
        cluster_points = point_cloud[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        print("z = ", str(centroid[2]))

    print(centroids)
    # Crear una nube de puntos para los centroides
    centroid_points = np.array(centroids)
    centroid_cloud = o3d.geometry.PointCloud()
    centroid_cloud.points = o3d.utility.Vector3dVector(centroid_points)

    # Asignar un color distintivo a cada centroide (por ejemplo, rojo)
    centroid_colors = np.tile([[1, 0, 0]], (len(centroids), 1))  # Rojo
    centroid_cloud.colors = o3d.utility.Vector3dVector(centroid_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    colors = np.ones_like(point_cloud) * [0, 0, 0]  # Blanco
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualizar la nube de puntos y los centroides
    o3d.visualization.draw_geometries([pcd, centroid_cloud])


# VISUALIZACION

# # Crear un objeto de nube de puntos Open3D
# pcd = o3d.geometry.PointCloud()

# # Asignar las posiciones de los puntos
# pcd.points = o3d.utility.Vector3dVector(point_cloud)

# # Asignar los colores (asegúrate de que están normalizados entre 0 y 1)
# pcd.colors = o3d.utility.Vector3dVector(colors / 255.0) # Asegúrate de que el color está normalizado

# # NEW
# o3d.io.write_point_cloud("./point_clouds/new_calibration.ply",pcd, print_progress= True)

# # OLD
# # o3d.io.write_point_cloud("./point_clouds/old_calibration.ply",pcd, print_progress= True)






# Visualizar
# o3d.visualization.draw_geometries([pcd])

# viewer = o3d.visualization.Visualizer()
# viewer.create_window()
# viewer.add_geometry(pcd)

# opt = viewer.get_render_option()
# opt.point_size = 10

# viewer.run()
# viewer.destroy_window()

