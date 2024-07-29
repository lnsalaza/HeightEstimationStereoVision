import os
import cv2
import csv
import json
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import keypoint_extraction as kp

from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Aplicar el filtro bilateral
sigma = 1.5  # Parámetro de sigma utilizado para el filtrado WLS.
lmbda = 8000.0  # Parámetro lambda usado en el filtrado WLS.

# Definición de los videos y matrices de configuración
configs = {
    'matlab_1': {
        'LEFT_VIDEO': '../videos/rectified/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/right_rectified.avi',
        'MATRIX_Q': '../config_files/newStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'opencv_1': {
        'LEFT_VIDEO': '../videos/rectified/left_rectified_old.avi',
        'RIGHT_VIDEO': '../videos/rectified/right_rectified_old.avi',
        'MATRIX_Q': '../config_files/old_config/stereoMap.xml',
        'disparity_to_depth_map': 'disparityToDepthMap',
        'numDisparities': 52,
        'blockSize': 10, 
        'minDisparity': 0,
        'disp12MaxDiff': 36,
        'uniquenessRatio': 39,
        'preFilterCap': 25,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'matlab_2': {
        'LEFT_VIDEO': '../videos/rectified/left_rectified_matlab_2.avi',
        'RIGHT_VIDEO': '../videos/rectified/right_rectified_matlab_2.avi',
        'MATRIX_Q': '../config_files/laser_config/including_Y_rotation_random/lyrrStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    }
}


situations = {
    '150_front': 60,
    '150_bodyside_variant': 150,
    '150_500': 3930,
    '200_front': 720,
    '200_shaking_hands_variant': 750,
    '200_400_front': 4080,
    '200_400_sitdown': 6150,
    '200_400_sitdown_side_variant': 6240,
    '250_front': 1020,
    '250_oneback_one_front_variant': 1050,
    '250_side_variant': 1140,
    '250_350': 4200,
    '250_500': 6900,
    '250_600': 4470,
    '250_600_perspective_variant': 4590,
    '300_front': 1290,
    '350_front': 1530,
    '350_side_variant': 1800,
    '400_front': 2010,
    '400_oneside_variant': 2130,
    '400_120cm_h_variant': 5160,
    '450_front': 2310,
    '450_side_variant': 2370,
    '450_600': 4710,
    '500_front': 2700,
    '500_oneside_variant': 2670,
    '550_front': 3000,
    '550_oneside_variant': 2940,
    '600_front': 3240,
    '600_oneside_variant': 3150
}

# situations = {
#     '150_front': 60,
#     '150_500': 3930,
# }


# Función para seleccionar configuración de cámara
def select_camera_config(camera_type):
    config = configs[camera_type]
    LEFT_VIDEO = config['LEFT_VIDEO']
    RIGHT_VIDEO = config['RIGHT_VIDEO']
    MATRIX_Q = config['MATRIX_Q']
    
    fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
    Q = fs.getNode(config['disparity_to_depth_map']).mat()
    fs.release()
    
    return LEFT_VIDEO, RIGHT_VIDEO, Q

# LEFT_VIDEO, RIGHT_VIDEO, Q = select_camera_config("new")


# Función para guardar la nube de puntos
def save_point_cloud(point_cloud, colors, camera_type, situation):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    filename = f"./point_clouds/{camera_type}_{situation}.ply"
    o3d.io.write_point_cloud(filename, pcd, print_progress=True)

# # --------------------------------------------------- KEYPOINTS EXTRACTION -------------------------------------------------------

# # Load a model
# model = YOLO('yolov8n-pose.pt').to(device=device)  # load an official model



# # Extract results
# def get_keypoints(source):
#     results = model(source=source, show=False, save = False, conf=0.85)[0] 
#     keypoints = np.array(results.keypoints.xy.cpu())
#     return keypoints

# def get_roi(source):
#     results = model(source=source, show=False, save = False, conf=0.85)[0] 
#     roi = np.array(results.boxes.xyxy.cpu())
#     return roi

# def apply_roi_mask(image, roi):
#     mask = np.zeros(image.shape[:2], dtype=np.uint8) 

#     # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
#     for coor in roi:
#         mask[int(coor[1]):int(coor[3]), int(coor[0]):int(coor[2])] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos

#     # Aplica la máscara a la imagen
#     masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)

#     return masked_image

# def apply_keypoints_mask(image, keypoints):
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     # Inicializa la máscara como una copia de la máscara original (normalmente toda en ceros)
#     for person in keypoints:
#         for kp in person:
#             y, x = int(kp[1]), int(kp[0])
#             # Verificar si las coordenadas están dentro de los límites de la imagen
#             if 0 <= y - 1 < image.shape[0] and 0 <= x - 1 < image.shape[1]:
#                 mask[y - 1, x - 1] = 1  # Pone en 1 los pixeles dentro de los cuadrados definidos
#     # Aplica la máscara a la imagen
#     masked_image = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8) * 255)
#     return masked_image


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
    new_image_filename = f'{image_name}_{next_number}.png'
    # Ruta completa del archivo
    full_path = os.path.join(path, new_image_filename)

    # Convertir a escala de grises si es necesario
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Guardar la imagen usando cv2.imwrite
    cv2.imwrite(full_path, image)



# --------------------------------------------------- DENSE POINT CLOUD ----------------------------------------------------------

def extract_frame(video_path, n_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    retval, frame = cap.read()
    cap.release()
    if not retval:
        raise ValueError(f"No se pudo leer el frame {n_frame}")
    return frame

# Función para extraer un frame específico de los videos izquierdo y derecho
def extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color=True, save=True):
    image_l = extract_frame(LEFT_VIDEO, n_frame)
    image_r = extract_frame(RIGHT_VIDEO, n_frame)

    if not color:
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    if save:
        cv2.imwrite(f"../images/image_l_{n_frame}.png", image_l)
        cv2.imwrite(f"../images/image_r_{n_frame}.png", image_r)
    
    return image_l, image_r

# Función para extraer frames según la situación y configuración de cámara
def extract_situation_frames(camera_type, situation, color=True, save=True):
    if situation in situations:
        n_frame = situations[situation]
        LEFT_VIDEO, RIGHT_VIDEO, Q = select_camera_config(camera_type)
        return extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color, save), Q
    else:
        raise ValueError("Situación no encontrada en el diccionario.")
    

def compute_disparity(left_image, right_image, camera_type):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    config = configs[camera_type]

    blockSize_var = config['blockSize']
    P1 = 8 * 3 * (blockSize_var ** 2)  
    P2 = 32 * 3 * (blockSize_var ** 2) 

    stereo = cv2.StereoSGBM_create(
        numDisparities = config['numDisparities'],
        blockSize = blockSize_var, 
        minDisparity=config['blockSize'],
        P1=P1,
        P2=P2,
        disp12MaxDiff=config['disp12MaxDiff'],
        uniquenessRatio=config['uniquenessRatio'],
        preFilterCap=config['preFilterCap'],
        mode=config['mode']
    )

    # Calcular el mapa de disparidad de la imagen izquierda a la derecha
    left_disp = stereo.compute(left_image, right_image)
    #.astype(np.float32) / 16.0

    # Crear el matcher derecho basado en el matcher izquierdo para consistencia
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Calcular el mapa de disparidad de la imagen derecha a la izquierda
    right_disp = right_matcher.compute(right_image, left_image)

    # Crear el filtro WLS y configurarlo
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Filtrar el mapa de disparidad utilizando el filtro WLS
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)

    # Normalización para la visualización o procesamiento posterior
    #filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    # filtered_disp = np.uint8(filtered_disp)
    return filtered_disp




def disparity_to_pointcloud(disparity, Q, image, custom_mask=None):
    points_3D = cv2.reprojectImageTo3D(disparity, Q) 
    mask = disparity > 0

    if custom_mask is not None:
        mask  = custom_mask > 0

    out_points = points_3D[mask]
    out_colors = image[mask]

    return out_points, out_colors


# CREACIÓN DE NUBE DE PUNTOS



def apply_dbscan(point_cloud, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
    labels = db.labels_
    return labels

def get_centroids(point_cloud, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if not unique_labels:
        print("No hay clusters.")
        return None
    else:
        centroids = []
        for label in unique_labels:
            cluster_points = point_cloud[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        return np.array(centroids)

def create_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return pcd

def save_point_cloud(point_cloud, colors, filename):
    pcd = create_point_cloud(point_cloud, colors)
    o3d.io.write_point_cloud(filename, pcd, print_progress=True)

def process_point_cloud(point_cloud, eps, min_samples, base_filename):
    labels = apply_dbscan(point_cloud, eps, min_samples)
    centroids = get_centroids(point_cloud, labels)

    if centroids is not None:
        original_cloud_colors = np.ones_like(point_cloud) * [0, 0, 255]  # Azul
        original_filename = f"{base_filename}_original.ply"
        save_point_cloud(point_cloud, original_cloud_colors, original_filename)

        centroid_colors = np.tile([[255, 0, 0]], (len(centroids), 1))  # Rojo
        centroid_filename = f"{base_filename}_centroids.ply"
        save_point_cloud(centroids, centroid_colors, centroid_filename)
    return centroids

def generate_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=True, ):
    
    if use_roi:
        roi = kp.get_roi(img_l)
        result_image = kp.apply_roi_mask(disparity, roi)
        save_image("../images/prediction_results/", result_image, "filtered_roi", False)
        eps, min_samples = 5, 2000
    else:
        keypoints = kp.get_keypoints(img_l)
        result_image = kp.apply_keypoints_mask(disparity, keypoints)
        save_image("../images/prediction_results/", result_image, "filtered_keypoints", False)

        eps = 50 if "matlab" in camera_type else 10
        min_samples = 6

    point_cloud, colors = disparity_to_pointcloud(disparity, Q, img_l, result_image)
    point_cloud = point_cloud.astype(np.float64)
    
    return point_cloud, colors, eps, min_samples

def roi_source_point_cloud(img_l, img_r, Q):
    eps, min_samples = 5, 1800

    roi_left = kp.get_roi(img_l)
    roi_right = kp.get_roi(img_r)

    result_img_left = kp.apply_roi_mask(img_l, roi_left)
    result_img_right = kp.apply_roi_mask(img_r, roi_right)

    disparity = compute_disparity(result_img_left, result_img_right)

    filtered_disparity = kp.apply_roi_mask(disparity, roi_left)

    dense_point_cloud, dense_colors = disparity_to_pointcloud(disparity, Q, img_l, filtered_disparity)

    return filtered_disparity, dense_point_cloud, dense_colors, eps, min_samples



def save_dense_point_cloud(point_cloud, colors, base_filename):
    dense_filename = f"{base_filename}_dense.ply"
    save_point_cloud(point_cloud, colors, dense_filename)

# def save_dataset(centroids):
#     data = []
#     true_z = true_z_values
###########################################################################################################
# # Flujo principal
# camera_type, situation = 'new', '150_front'
# (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
# img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
# img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

# disparity = compute_disparity(img_l, img_r)

# # with open("../config_files/stereoParameters.json", "r") as file:
# #     params = json.load(file)
# #     baseline = -(params["stereoT"][0])
# #     fpx = params["flCamera1"][0]

# # Generar nube de puntos densa sin filtrado adicional
# dense_point_cloud, dense_colors = disparity_to_pointcloud(disparity, Q, img_l)
# dense_point_cloud = dense_point_cloud.astype(np.float64)
# base_filename = f"./point_clouds/{camera_type}_calibration_{situation}"
# save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

# # Generar nube de puntos con filtrado y aplicar DBSCAN
# point_cloud, colors, eps, min_samples = generate_filtered_point_cloud(img_l, disparity, Q, use_roi=False)
# process_point_cloud(point_cloud, eps, min_samples, base_filename)
###############################################################################################################




# Flujo principal para todas las situaciones
data = []
camera_type = 'opencv_1'
mask_type = 'keypoint'
is_roi = (mask_type == "roi")

for situation in situations:
    try:
        print(f"\nProcesando situación: {situation}")
        (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        
        #disparity, point_cloud, colors, eps, min_samples = roi_source_point_cloud(img_l, img_r, Q)
        
        disparity = compute_disparity(img_l, img_r, camera_type)

        # # Generar nube de puntos densa sin filtrado adicional
        dense_point_cloud, dense_colors = disparity_to_pointcloud(disparity, Q, img_l)
        dense_point_cloud = dense_point_cloud.astype(np.float64)

        base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}"
        if not os.path.exists(os.path.dirname(base_filename)):
            os.makedirs(os.path.dirname(base_filename))

        # base_filename = f"./point_clouds/test_old/{camera_type}_{situation}"
        save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

        # # Generar nube de puntos con filtrado y aplicar DBSCAN
        point_cloud, colors, eps, min_samples = generate_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=is_roi)
        centroids = process_point_cloud(point_cloud, eps, min_samples, base_filename)

        z_estimations = [centroid[2] for centroid in centroids] if centroids is not None else []
        data.append({
            "situation": situation,
            **{f"z_estimation_{i+1}": z for i, z in enumerate(z_estimations)}
        })

        
    except Exception as e:
        print(f"Error procesando {situation}: {e}")

# Guardar dataset como CSV
dataset_path = f"../datasets/data/z_estimation_{camera_type}_{mask_type}.csv"
# dataset_path = f"../datasets/data/z_estimation_{camera_type}_keypoints_no_astype_no_norm.csv"
# dataset_path = f"../datasets/data/z_estimation_{camera_type}_roi.csv"
# dataset_path = f"../datasets/data/z_estimation_{camera_type}_roi_before_disparity.csv"

if not os.path.exists(os.path.dirname(dataset_path)):
    os.makedirs(os.path.dirname(dataset_path))

max_z_count = max(len(row) - 1 for row in data) # -1 porque situation no es una columna z_estimation

fieldnames = ["situation"] + [f"z_estimation_{i+1}" for i in range(max_z_count)]

with open(dataset_path, "w", newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in data:
        writer.writerow(row)
print(f"Dataset guardado en {dataset_path}")
