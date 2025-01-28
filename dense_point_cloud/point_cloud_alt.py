import json
import os
import cv2
import numpy as np

# Importamos la versión estereoscópica (cálculo de disparidad, reprojectImageTo3D, etc.)
from dense_point_cloud.pc_generation_realsense import DotDict
import dense_point_cloud.pc_generation as pcGen

# Importamos la versión "ML" (RAFT, SELECTIVE, etc. con disparidad)
import dense_point_cloud.pc_generation_ML as pcGen_ML
import dense_point_cloud.pc_generation_realsense as pcGen_rs
# Este sería tu módulo para RealSense (si ya lo tienes). 
# O la lógica que hayas adaptado en 'pc_generation_realsense.py':
# import dense_point_cloud.pc_generation_realsense as pcGen_rs

from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

from dense_point_cloud.util import (
    prepare_point_cloud,
    prepare_individual_point_clouds,
    filter_points_by_optimal_range,
    get_Y_bounds,
    get_max_coordinates
)

from dense_point_cloud.Selective_IGEV.bridge_selective import get_SELECTIVE_disparity_map
from dense_point_cloud.RAFTStereo.bridge_raft import get_RAFT_disparity_map

from calibration.rectification import load_stereo_maps 
from dense_point_cloud.features_script import *
from dense_point_cloud.FaceHeightEstimation.height_stimation import (
    compute_height_using_face_metrics, 
    compute_separation_eyes_camera
)

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


##############################################################################
#               Clase para normalizar (escalar) y corregir nubes            #
##############################################################################
class PointCloudScaler:
    def __init__(self, reference_point, scale_factor):
        self.reference_point = np.array(reference_point, dtype=np.float32)
        self.scale_factor = scale_factor

    def calculate_scaled_positions(self, points):
        shifted_points = points - self.reference_point
        scaled_points = self.scale_factor * shifted_points
        new_positions = scaled_points + self.reference_point
        return new_positions

    def scale_cloud(self, points):
        return self.calculate_scaled_positions(points)

def correct_depth_o3d(points, alpha=1.0):
    """
    Aplica una corrección de profundidad usando una potencia en Z.
    Si alpha=1.0 => No hay deformación.

    Si tu Realsense (o la calibración estéreo) ya está correcta y la altura
    de los objetos se ve bien, puedes poner alpha=1.0. 
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

    # Evita divisiones por cero:
    Z_safe = np.where(Z == 0, np.finfo(float).eps, Z)

    # Corrección de potencia:
    Z_corrected = Z_safe ** alpha

    # Ajuste X, Y en la misma proporción que cambió Z:
    factor = Z_corrected / Z_safe
    X_corrected = X * factor
    Y_corrected = Y * factor

    # ----------------------------------------------------------
    # Si YA NO necesitas ajustar la escala ni offsets en Z:
    # Z_corrected_final = Z_corrected
    #
    # Si requieres retocar un poco la profundidad,
    # podrías descomentar lo siguiente y ajustar a,b:
    #
    # a = 1.0
    # b = 0.0
    # Z_corrected_final = a * Z_corrected + b
    #
    # ----------------------------------------------------------
    Z_corrected_final = Z  # Sin offset ni factor lineal extra

    corrected_points = np.vstack((X_corrected, Y_corrected, Z_corrected_final)).T
    return corrected_points

def process_numpy_point_cloud(points_np, reference_point=[0, 0, 0], scale_factor=1, alpha = 0.9665):
    scaler = PointCloudScaler(reference_point=reference_point, scale_factor=scale_factor)
    scaled_points_np = scaler.scale_cloud(points_np)
    corrected_points_np = correct_depth_o3d(scaled_points_np, 0.9665)
    return corrected_points_np


##############################################################################
#                           Funciones de rectificación                       #
##############################################################################
def rectify_images(img_0: np.array, img_1: np.array, config: str):
    """
    Rectifica un par de imágenes estéreo usando los mapas de rectificación 
    correspondientes al perfil de calibración dado.
    """
    # Ajusta la ruta según tu proyecto
    map_path = f'config_files/{config}/stereo_map.xml'
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"No se encontró el archivo de mapa de rectificación: {map_path}")
    
    stereo_maps = load_stereo_maps(map_path)
    
    # Aplica los mapas de rectificación
    img_0_rect = cv2.remap(img_0, stereo_maps['Left'][0], stereo_maps['Left'][1], cv2.INTER_LINEAR)
    img_1_rect = cv2.remap(img_1, stereo_maps['Right'][0], stereo_maps['Right'][1], cv2.INTER_LINEAR)

    # Convierte a RGB si quieres
    img_0_rect = cv2.cvtColor(img_0_rect, cv2.COLOR_BGR2RGB)
    img_1_rect = cv2.cvtColor(img_1_rect, cv2.COLOR_BGR2RGB)
    return img_0_rect, img_1_rect


##############################################################################
#                        Función para obtener disparidad                     #
##############################################################################
def compute_disparity(img_0: np.array, img_1: np.array, config: dict, method: str):
    """
    Calcula el mapa de disparidad de un par de imágenes usando el método especificado.
    """
    methods_config = config['disparity_methods']

    if method == 'SGBM' and methods_config['SGBM']['enabled']:
        params = methods_config['SGBM']['params']
        disparity = pcGen.compute_disparity(img_0, img_1, params)

    elif method == 'WLS-SGBM' and methods_config['WLS-SGBM']['enabled']:
        params = methods_config['WLS-SGBM']['params']
        disparity = pcGen.compute_disparity(img_0, img_1, params)

    elif method == 'RAFT' and methods_config['RAFT']['enabled']:
        disparity = get_RAFT_disparity_map(
            restore_ckpt=methods_config['RAFT']['params']['restore_ckpt'],
            img_left_array=img_0,
            img_right_array=img_1,
            save_numpy=True, 
            slow_fast_gru=True,
        )
        disparity = disparity[0]

    elif method == 'SELECTIVE' and methods_config['SELECTIVE']['enabled']:
        disparity = get_SELECTIVE_disparity_map(
            restore_ckpt=methods_config['SELECTIVE']['params']['restore_ckpt'],
            img_left_array=img_0,
            img_right_array=img_1,
            save_numpy=True,
            slow_fast_gru=True,
        )
        disparity = disparity[0]

    else:
        raise ValueError(f"The disparity method '{method}' is not enabled or does not exist.")
    return disparity


##############################################################################
#              Generación de nubes de puntos: estéreo vs RealSense           #
##############################################################################
def generate_dense_point_cloud(
    img_0: np.array, 
    img_1: np.array, 
    config: dict, 
    method: str, 
    use_max_disparity: bool, 
    normalize: bool = True
):
    """
    Genera una nube de puntos 3D densa. 

    - Si 'method' es uno de ['SGBM','WLS-SGBM','RAFT','SELECTIVE'], 
      se asume que img_0=left, img_1=right, y se calcula la disparidad estéreo.

    - Si 'method' == 'realsense', se asume que img_0=color, img_1=depth, 
      y se reproyecta usando intrínsecos en config['camera_params'].

    :param img_0: Puede ser la imagen izquierda (estéreo) o la de color (RealSense).
    :param img_1: Puede ser la imagen derecha (estéreo) o la de profundidad (RealSense).
    """
    camera_params = config['camera_params']

    if method in ['SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE']:
        # ------ Estéreo: calcular disparidad ------
        disparity_map = compute_disparity(img_0, img_1, config, method)

        Q = np.array(camera_params['Q_matrix'])
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx1 = camera_params['cx1']
        cx2 = camera_params['cx2']
        cy = camera_params['cy']
        baseline = camera_params['baseline']

        # Escoger la función de reproyección según sea SGBM o RAFT
        if method in ['SGBM', 'WLS-SGBM']:
            # pc_generation.py
            point_cloud, colors = pcGen.disparity_to_pointcloud(
                disparity_map, Q, img_0, use_max_disparity=use_max_disparity
            )
            scale_factor = 1.87
        else:
            # RAFT / SELECTIVE => pc_generation_ML.py
            point_cloud, colors = pcGen_ML.disparity_to_pointcloud(
                disparity_map, fx, fy, cx1, cx2, cy, baseline, img_0, use_max_disparity
            )
            scale_factor = 0.150005

    elif method == 'realsense':
        # ------ RealSense: ya tenemos color + depth ------
        # Supongamos que pc_generation_realsense.py (o similar) 
        # nos da la función depth_to_pointcloud(...) 
        # Si no, incluyemos la lógica aquí. Ejemplo simplificado:

        depth_intrinsics = DotDict(
        depth_scale=0.0010000000474974513,
        fx=631.189453125,
        fy=631.189453125,
        ppx=647.0123901367188,
        ppy=362.94287109375
        )
        fx = depth_intrinsics.fx
        fy = depth_intrinsics.fy
        ppx = depth_intrinsics.ppx
        ppy = depth_intrinsics.ppy
        depth_scale = depth_intrinsics.depth_scale

        print(f"fx: {fx}, fy: {fy}, ppx: {ppx}, ppy: {ppy}, depth_scale: {depth_scale}")

        # Convertir la profundidad y el color en nube de puntos
        # Lógica simplificada de realsense:
        # (Asume que img_1 = depth, img_0 = color)
        import dense_point_cloud.pc_generation_realsense as pcGen_rs

        point_cloud, colors = pcGen_rs.depth_to_pointcloud(
            depth_image=img_1,
            color_image=img_0,
            intrinsics=pcGen_rs.DotDict(
                fx=fx, fy=fy, ppx=ppx, ppy=ppy, depth_scale=depth_scale
            ),
            custom_mask=None,
            is_video=False,
            to_unit="mm"
        )
        scale_factor = 1.0  # Ajusta si quieres otra normalización

    else:
        raise ValueError(f"Unknown method '{method}'. Use stereo methods or 'realsense'.")

    # Normalizar la nube de puntos si se solicita
    if normalize:
        point_cloud = process_numpy_point_cloud(point_cloud, scale_factor=scale_factor, alpha=1.0005119)

    # Prepara la nube para uso (opcional: remover outliers, etc.)
    prepare_point_cloud(point_cloud, colors)

    return point_cloud, colors


##############################################################################
#   Corrección adicional a la nube (con modelo de regresión u otra lógica)   #
##############################################################################
def point_cloud_correction(point_cloud: np.array, model: any) -> np.array:
    return pcGen.point_cloud_correction(point_cloud, model)


##############################################################################
#         Generar nubes filtradas combinadas (ROI/keypoints), estéreo        #
##############################################################################
def generate_combined_filtered_point_cloud(
    img_0: np.array, 
    img_1: np.array, 
    config: dict, 
    method: str, 
    use_roi: bool, 
    use_max_disparity: bool, 
    normalize: bool = True
):
    """
    Genera una nube de puntos 3D filtrada, combinando objetos/personas en una sola nube.
    Funciona principalmente para estéreo (métodos 'SGBM','WLS-SGBM','RAFT','SELECTIVE').
    """
    # Enfoque: primero obtienes la disparidad, luego llamas a
    #   pcGen.generate_all_filtered_point_cloud o pcGen_ML.generate_all_filtered_point_cloud
    # Dependiendo del método (estéreo).
    # NO cubre 'realsense' directamente. Si quieres filtrar RealSense,
    # tendrás que adaptar tu módulo pc_generation_realsense.
    disparity_map = compute_disparity(img_0, img_1, config, method)

    camera_params = config['camera_params']
    Q = np.array(camera_params['Q_matrix'])
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx1 = camera_params['cx1']
    cx2 = camera_params['cx2']
    cy = camera_params['cy']
    baseline = camera_params['baseline']

    if method in ['SGBM', 'WLS-SGBM']:
        point_cloud, colors, eps, min_samples = pcGen.generate_all_filtered_point_cloud(
            img_0, disparity_map, Q, camera_type="matlab", 
            use_roi=use_roi, use_max_disparity=use_max_disparity
        )
        scale_factor = 1.87
    else:
        point_cloud, colors, eps, min_samples = pcGen_ML.generate_all_filtered_point_cloud(
            img_0, disparity_map, fx, fy, cx1, cx2, cy, baseline, 
            camera_type="matlab", use_roi=use_roi, use_max_disparity=use_max_disparity
        )
        scale_factor = 0.150005

    if normalize:
        point_cloud = process_numpy_point_cloud(point_cloud, scale_factor=scale_factor, alpha=1.0005119)
    prepare_point_cloud(point_cloud, colors)

    return point_cloud, colors


import dense_point_cloud.pc_generation as pcGen
import dense_point_cloud.pc_generation_ML as pcGen_ML
import dense_point_cloud.pc_generation_realsense as pcGen_rs

def generate_individual_filtered_point_clouds(
    img_0: np.array, 
    img_1: np.array, 
    config: dict, 
    method: str, 
    use_roi: bool, 
    use_max_disparity: bool, 
    normalize: bool = True
):
    """
    Genera listas separadas de nubes de puntos, colores y keypoints 3D para cada
    objeto/persona detectada, tanto para métodos estéreo (SGBM, WLS-SGBM, RAFT, SELECTIVE)
    como para RealSense (method="realsense").

    - Si method es uno de ['SGBM','WLS-SGBM','RAFT','SELECTIVE'], entonces:
        img_0 = imagen izquierda
        img_1 = imagen derecha
        Se calcula la disparidad y luego se generan las nubes filtradas.

    - Si method == "realsense", entonces:
        img_0 = imagen de color
        img_1 = imagen de profundidad (16 bits, cargada con cv2.IMREAD_UNCHANGED)
        Se generan las nubes filtradas directamente a partir de la depth.

    Retorna:
        (point_cloud_list, color_list, keypoints3d_list, max_coords)
        donde cada elemento en las listas corresponde a un objeto detectado.
    """
    camera_params = config['camera_params']
    scale_factor = 1
    # -----------------------------------------------------------------------
    # CASO 1: Flujo Estéreo (cálculo de disparidad + reproyección)
    # -----------------------------------------------------------------------
    if method in ['SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE']:
        # 1) Obtener disparidad
        disparity_map = compute_disparity(img_0, img_1, config, method)

        # 2) Extraer parámetros
        Q = np.array(camera_params['Q_matrix'])
        fx = camera_params['fx']
        fy = camera_params['fy']
        cx1 = camera_params['cx1']
        cx2 = camera_params['cx2']
        cy = camera_params['cy']
        baseline = camera_params['baseline']

        # 3) Generar las nubes filtradas, ROI/keypoints
        if method in ['SGBM', 'WLS-SGBM']:
            # Usa las funciones de pc_generation.py
            (
                point_cloud_list, 
                color_list, 
                eps, 
                min_samples, 
                keypoints3d_list
            ) = pcGen.generate_filtered_point_cloud(
                img_l=img_0,               # "left" => img_0
                disparity=disparity_map,
                Q=Q,
                camera_type="matlab",      # nomenclatura que usabas internamente
                use_roi=use_roi,
                use_max_disparity=use_max_disparity
            )
            

        else:
            # RAFT, SELECTIVE => pc_generation_ML.py
            (
                point_cloud_list, 
                color_list, 
                eps, 
                min_samples, 
                keypoints3d_list
            ) = pcGen_ML.generate_filtered_point_cloud(
                img_l=img_0,               # "left" => img_0
                disparity=disparity_map,
                fx=fx,
                fy=fy,
                cx1=cx1,
                cx2=cx2,
                cy=cy,
                baseline=baseline,
                camera_type="matlab",
                use_roi=use_roi,
                use_max_disparity=use_max_disparity
            )
            

    # -----------------------------------------------------------------------
    # CASO 2: Flujo RealSense (imagen color + depth map + reproyección)
    # -----------------------------------------------------------------------
    elif method == 'realsense':
        # 1) Extraer intrínsecos de config
        depth_intrinsics = DotDict(
        depth_scale=0.0010000000474974513,
        fx=631.189453125,
        fy=631.189453125,
        ppx=647.0123901367188,
        ppy=362.94287109375
        )
        fx = depth_intrinsics.fx
        fy = depth_intrinsics.fy
        ppx = depth_intrinsics.ppx
        ppy = depth_intrinsics.ppy
        depth_scale = depth_intrinsics.depth_scale
        # 2) Dependiendo de cómo segmentes, "disparity" pasa a ser "depth"
        #    Ten en cuenta que tus funciones en keypoint_extraction.py
        #    pueden requerir adaptaciones para mascar la "depth_image".
        depth_image = img_1  # Asumimos que img_1 es la imagen de profundidad

        (
                point_cloud_list, 
                color_list, 
                eps, 
                min_samples, 
                keypoints3d_list
            ) = pcGen_rs.generate_filtered_point_cloud(
                color_image=img_0,
                depth_image=depth_image,
                intrinsics=depth_intrinsics,
                camera_type="matlab",
                use_roi=use_roi,
            )
    else:
        raise ValueError(f"Unknown method '{method}'. "
                         f"Use stereo methods or 'realsense'.")

    # -----------------------------------------------------------------------
    # Normalizar las nubes si se solicitó
    # -----------------------------------------------------------------------
    if normalize:
        print(f"Normalizando con scale_factor = {scale_factor} ...")
        normalized_pcs = [
            process_numpy_point_cloud(cloud, scale_factor=scale_factor, alpha = 0.9665)
            for cloud in point_cloud_list
        ]
        # Si tu lista de keypoints 3D existe:
        normalized_kpts = []
        if keypoints3d_list:
            normalized_kpts = [
                process_numpy_point_cloud(kps, scale_factor=scale_factor, alpha = 0.9665)
                for kps in keypoints3d_list
            ]
        else:
            normalized_kpts = []

        prepare_individual_point_clouds(normalized_pcs, color_list, normalized_kpts)
        max_coords = get_max_coordinates(normalized_pcs)

        return normalized_pcs, color_list, normalized_kpts, max_coords
    else:
        # No normalizar
        prepare_individual_point_clouds(point_cloud_list, color_list, keypoints3d_list)
        max_coords = get_max_coordinates(point_cloud_list)
        return point_cloud_list, color_list, keypoints3d_list, max_coords

def get_features(keypoints):
    list_heights = []
    list_tronco_normal = []
    list_centroides = []
    list_head_normal = []
    list_is_centroid_to_nariz = []
    centroide = np.array([])
    avg_normal = np.array([0, 0, 0])

    avg_normal_head = np.array([0, 0, 0])
    list_union_centroids = []
    avg_head_centroid = np.array([0, 0, 0])
    character = ""
    confianza = 0

    # Get Height
    for person in keypoints:
        # funcion de altura estimate_height_from_point_cloud
        estimated_height, _centroid = estimate_height_from_point_cloud(point_cloud=person, m_initial=100)
        list_heights.append(estimated_height)

    kps_filtered = np.array(keypoints)[:, [0, 1, 2, 5, 6, 11, 12], :]

    # Get each point of person, all person
    list_points_persons, list_ponits_bodies_nofiltered = get_each_point_of_person(kps_filtered)

    # Get centroid and normal
    get_centroid_and_normal(list_points_persons, list_ponits_bodies_nofiltered, list_centroides, list_tronco_normal, list_head_normal, list_is_centroid_to_nariz)

    if len(list_centroides) > 0:
        # Centroide grupal (centroide del grupo)
        centroide =  np.mean(np.array(list_centroides), axis=0)
        ## Vector promedio del tronco
        avg_normal = average_normals(list_tronco_normal) 
        if avg_normal is not None:
            avg_normal_head, list_union_centroids, avg_head_centroid, character, confianza = get_group_features(list_centroides, centroide, avg_normal, list_head_normal, list_points_persons)

    return get_structure_data(keypoints, character, confianza, list_tronco_normal, list_head_normal, avg_normal, avg_normal_head, list_centroides, list_union_centroids, centroide, avg_head_centroid, list_is_centroid_to_nariz, list_heights)


def generate_filtered_point_cloud_with_features(
    img_0: np.array, 
    img_1: np.array, 
    config: dict, 
    method: str, 
    use_roi: bool, 
    use_max_disparity: bool, 
    normalize: bool = True
):
    """
    Similar a generate_individual_filtered_point_clouds, pero además extrae 
    características (por ejemplo, con 'features_script.py').
    """
    
    camera_params = config['camera_params']
    scale_factor = 1
    Q = np.array(camera_params['Q_matrix'])
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx1 = camera_params['cx1']
    cx2 = camera_params['cx2']
    cy = camera_params['cy']
    baseline = camera_params['baseline']

    if method in ['SGBM', 'WLS-SGBM']:
        disparity_map = compute_disparity(img_0, img_1, config, method)

        (
            point_cloud_list, 
            color_list, 
            eps, 
            min_samples, 
            keypoints3d_list
        ) = pcGen.generate_filtered_point_cloud(
            img_0, disparity_map, Q, "matlab", use_roi, use_max_disparity
        )
        scale_factor = 1.87
    if method in ['RAFT', 'SELECTIVE']:
        disparity_map = compute_disparity(img_0, img_1, config, method)
        (
            point_cloud_list, 
            color_list, 
            eps, 
            min_samples, 
            keypoints3d_list
        ) = pcGen_ML.generate_filtered_point_cloud(
            img_0, disparity_map, fx, fy, cx1, cx2, cy, baseline, 
            "matlab", use_roi, use_max_disparity
        )
        scale_factor = 0.150005
    else:
        # 1) Extraer intrínsecos de config
        depth_intrinsics = DotDict(
        depth_scale=0.0010000000474974513,
        fx=631.189453125,
        fy=631.189453125,
        ppx=647.0123901367188,
        ppy=362.94287109375
        )
        fx = depth_intrinsics.fx
        fy = depth_intrinsics.fy
        ppx = depth_intrinsics.ppx
        ppy = depth_intrinsics.ppy
        depth_scale = depth_intrinsics.depth_scale

        # 2) Generar nube de puntos
        depth_image = img_1  # Asumimos que img_1 es la imagen de profundidad

        (
                point_cloud_list, 
                color_list, 
                eps, 
                min_samples, 
                keypoints3d_list
            ) = pcGen_rs.generate_filtered_point_cloud(
                color_image=img_0,
                depth_image=depth_image,
                intrinsics=depth_intrinsics,
                camera_type="matlab",
                use_roi=use_roi,
            )
        scale_factor = 1

    if normalize:
        
        normalized_pcs = [process_numpy_point_cloud(cloud, scale_factor=scale_factor, alpha=1.0005119)
                          for cloud in point_cloud_list]
        normalized_kpts = [process_numpy_point_cloud(kps, scale_factor=scale_factor, alpha=1.0005119)
                           for kps in keypoints3d_list]

        prepare_individual_point_clouds(normalized_pcs, color_list, normalized_kpts)
        max_coords = get_max_coordinates(normalized_pcs)

        # Extraer características
        features = get_features(normalized_kpts)
        return normalized_pcs, color_list, normalized_kpts, features, max_coords

    else:
        prepare_individual_point_clouds(point_cloud_list, color_list, keypoints3d_list)
        max_coords = get_max_coordinates(point_cloud_list)

        features = get_features(keypoints3d_list)
        return point_cloud_list, color_list, keypoints3d_list, features, max_coords


##############################################################################
#        Funciones de altura con FaceHeightEstimation (estéreo)              #
##############################################################################
def compute_centroid(points, k=5, threshold_factor=1.0):
    if len(points) < k + 1:
        raise ValueError("La nube de puntos es demasiado pequeña para el valor de k.")
    tree = KDTree(points)
    noise_mask = np.zeros(points.shape[0], dtype=bool)

    distances, _ = tree.query(points, k=k+1)
    avg_distance = np.mean(distances[:, 1:])
    threshold = threshold_factor * avg_distance

    for i, point in enumerate(points):
        dists, indices = tree.query(point, k=k+1)
        dists = dists[1:]
        mean_depth = np.mean(dists)
        if np.any(np.abs(dists - mean_depth) > threshold):
            noise_mask[i] = True

    filtered_points = points[~noise_mask]
    if len(filtered_points) == 0:
        raise ValueError("Todos los puntos fueron considerados ruido.")
    centroid = np.mean(filtered_points, axis=0)
    return centroid

def compute_centroids(points, k=5, threshold_factor=1.0, eps_factor=2, min_samples=10):
    if len(points) < k + 1:
        raise ValueError("La nube de puntos es demasiado pequeña para el valor de k.")

    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)
    avg_distance = np.mean(distances[:, 1:])
    threshold = threshold_factor * avg_distance
    eps = eps_factor * avg_distance

    noise_mask = np.zeros(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        dists, indices = tree.query(point, k=k+1)
        dists = dists[1:]
        mean_depth = np.mean(dists)
        if np.any(np.abs(dists - mean_depth) > threshold):
            noise_mask[i] = True

    filtered_points = points[~noise_mask]
    if len(filtered_points) == 0:
        raise ValueError("Todos los puntos fueron considerados ruido.")

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = clustering.labels_

    centroids = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = filtered_points[labels == label]
        centroids.append(np.mean(cluster_points, axis=0))

    return centroids

def estimate_height_from_point_cloud(point_cloud: np.array, k: int = 5, threshold_factor: float = 1.0, m_initial: float = 50.0):
    try:
        centroid = compute_centroid(point_cloud, k=k, threshold_factor=threshold_factor)
        filtered_points = filter_points_by_optimal_range(point_cloud, centroid, m_initial)
        y_min, y_max = get_Y_bounds(filtered_points)
        if y_min is not None and y_max is not None:
            height = abs(y_max - y_min)
            print(f"Altura estimada: {height}")
            return height, centroid
        else:
            print("No se encontraron puntos en el rango óptimo.")
            return None, centroid
    except ValueError as ve:
        print(f"Error en compute_centroid: {ve}")
        return None, None

def estimate_height_from_face_proportions(img_0, img_1, config):
    """
    Usa la lógica de FaceHeightEstimation sólo para cámaras estéreo.
    """
    camera_config = config['camera_params']
    height, depth = compute_height_using_face_metrics(
        img_left=img_0, 
        img_right=img_1, 
        baseline=camera_config['baseline'], 
        fx=camera_config['fx'], 
        camera_center_left=[camera_config['cx1'], camera_config['cy']]
    )
    return height, depth

def estimate_separation_eyes_camera(img_0, img_1, config):
    camera_config = config['camera_params']
    height, depth = compute_separation_eyes_camera(
        img_left=img_0,
        img_right=img_1,
        baseline=camera_config['baseline'],
        fx=camera_config['fx'],
        camera_center_left=[camera_config['cx1'], camera_config['cy']]
    )
    return height, depth
