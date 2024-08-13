import json
import os
import cv2
import numpy as np
import dense_point_cloud.pc_generation as pcGen
import dense_point_cloud.pc_generation_ML as pcGen_ML
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from dense_point_cloud.util import prepare_point_cloud, prepare_individual_point_clouds, filter_points_by_optimal_range, get_Y_bounds
from dense_point_cloud.Selective_IGEV.bridge_selective import get_SELECTIVE_disparity_map
from dense_point_cloud.RAFTStereo.bridge_raft import get_RAFT_disparity_map
from calibration.rectification import load_stereo_maps 

from scipy.spatial import cKDTree


# Clase encargada de la normalizacion estandar de las nubes de puntos
class PointCloudNormalizer:
    def __init__(self, target_unit_scale=1.0):
        self.target_unit_scale = target_unit_scale

    def normalize(self, cloud):
        try:
            # Obtener los puntos como un numpy array
            points = np.asarray(cloud.points)
            
            # Crear un KD-Tree para buscar los vecinos más cercanos
            kdtree = cKDTree(points)
            
            # Buscar el vecino más cercano para cada punto
            distances, _ = kdtree.query(points, k=2)  # k=2 porque la primera distancia es 0 (el mismo punto)
            
            # Excluir distancias que sean 0 antes de encontrar el mínimo
            non_zero_distances = distances[:, 1][distances[:, 1] > 0]
            
            if non_zero_distances.size == 0:
                raise ValueError("All non-zero distances are zero. The point cloud might be degenerate.")
            
            # Tomar la distancia mínima que no sea cero
            min_dist = np.min(non_zero_distances)
            
            # Determinar el factor de escala para ajustar la nube a la escala deseada
            scale_factor = self.target_unit_scale / min_dist
            
            # Escalar la nube de puntos
            cloud.scale(scale_factor, center=cloud.get_center())

            # Mover la nube al origen
            cloud.translate(-cloud.get_center())
            
            return cloud
        except Exception as e:
            print(f"Error normalizing point cloud: {e}")
            return cloud  # Devuelve la nube original en caso de error

def process_numpy_point_cloud(points_np):
    """
    Normaliza la dimension de la nube de puntos ingresada a un tamaño estandar en donde no se pierden las distancias relativas entre objetos dentro de la nube 3D.

    Args:
        points_np (np.array): Array con los puntos de la nube de puntos 3D.

    Returns:
        return: Array con los puntos de la nube de puntos 3D normalizados.
    """
    # Convertir numpy array a Open3D PointCloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_np)
    
    # Normalizar la nube de puntos manteniendo la relación interna
    normalizer = PointCloudNormalizer(target_unit_scale=1.0)  # 1 unidad = 1 metro
    normalized_cloud = normalizer.normalize(cloud)
    
    # Convertir de nuevo a numpy array si es necesario
    normalized_points_np = np.asarray(normalized_cloud.points)
    
    return normalized_points_np

def rectify_images(img_left: np.array, img_right: np.array, config: str):
    """
    Rectifica un par de imágenes estéreo usando los mapas de rectificación correspondientes al perfil de calibración dado.

    Args:
        img_left (np.array): Imagen izquierda como array de numpy.
        img_right (np.array): Imagen derecha como array de numpy.
        profile_name (str): Nombre del perfil que contiene los mapas de rectificación.

    Returns:
        tuple: Tupla que contiene las imágenes izquierda y derecha rectificadas.
    """
    # Carga los mapas de rectificación desde el archivo XML asociado al perfil
    map_path = f'config_files/{config}/stereo_map.xml'
    if not os.path.exists(map_path):
        raise FileNotFoundError("No se encontró el archivo de mapa de rectificación para el perfil especificado.")
    
    stereo_maps = load_stereo_maps(map_path)
    
    # Aplica los mapas de rectificación
    img_left_rect = cv2.remap(img_left, stereo_maps['Left'][0], stereo_maps['Left'][1], cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, stereo_maps['Right'][0], stereo_maps['Right'][1], cv2.INTER_LINEAR)

    img_left_rect = cv2.cvtColor(img_left_rect, cv2.COLOR_BGR2RGB)
    img_right_rect = cv2.cvtColor(img_right_rect, cv2.COLOR_BGR2RGB)
    return img_left_rect, img_right_rect

def compute_disparity(img_left: np.array, img_right: np.array, config: dict, method: str):
    """
    Calcula el mapa de disparidad de un par de imágenes usando el método especificado en la configuración.
    
    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :return: Mapa de disparidad como array de numpy.
    """
    # Acceso a los métodos de disparidad configurados
    methods_config = config['disparity_methods']
    
    if method == 'SGBM' and methods_config['SGBM']['enabled']:
        params = methods_config['SGBM']['params']
        disparity = pcGen.compute_disparity(img_left, img_right, params)

    elif method == 'RAFT' and methods_config['RAFT']['enabled']:
        disparity = get_RAFT_disparity_map(
            restore_ckpt=methods_config['RAFT']['params']['restore_ckpt'],
            img_left_array=img_left,
            img_right_array=img_right,
            save_numpy=True, 
            slow_fast_gru=True, #--slow_fast_gru: True=GRUs de baja resolución iteran más frecuentemente (captura cambios rápidos, más tiempo de computación), 
                                #                 False(Default)=frecuencia estándar (suficiente para muchas aplicaciones, más eficiente).
        )
        disparity = disparity[0]
    elif method == 'SELECTIVE' and methods_config['SELECTIVE']['enabled']:
        # Usar Selective para calcular disparidad
        disparity = get_SELECTIVE_disparity_map(
            restore_ckpt=methods_config['SELECTIVE']['params']['restore_ckpt'],
            img_left_array=img_left,
            img_right_array=img_right,
            save_numpy=True,
            slow_fast_gru=True,
        )
        disparity = disparity[0]
    else:
        raise ValueError(f"The disparity method {method} is either not enabled or does not exist in the configuration.")

    return disparity


def generate_dense_point_cloud(img_left: np.array, img_right: np.array, config: dict, method: str, use_max_disparity: bool, normalize: bool = True):
    """
    Genera una nube de puntos 3D densa a partir de un par de imágenes estéreo utilizando el método especificado.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
    :param normalize: Booleano que indica si se debe normalizar la nube de puntos a una escala de unidad estándar.
    :return: Tuple que contiene la nube de puntos 3D densa y el array de colores correspondiente.
    """
    # Calcula el mapa de disparidad utilizando la función adecuada
    disparity_map = compute_disparity(img_left, img_right, config, method)

    # Acceder a parámetros relevantes desde la configuración
    Q = np.array(config['camera_params']['Q_matrix'])
    fx = config['camera_params']['fx']
    fy = config['camera_params']['fy']
    cx1 = config['camera_params']['cx1']
    cx2 = config['camera_params']['cx2']
    cy = config['camera_params']['cy']
    baseline = config['camera_params']['baseline']

    # Generar nube de puntos 3D densa
    if method == 'SGBM':
        point_cloud, colors = pcGen.disparity_to_pointcloud(disparity_map, Q, img_left, use_max_disparity=use_max_disparity)
    else:
        # Asumimos que RAFT y SELECTIVE usan la versión ML para reproyección
        point_cloud, colors = pcGen_ML.disparity_to_pointcloud(disparity_map, fx, fy, cx1, cx2, cy, baseline, img_left, use_max_disparity=use_max_disparity)

    # Normalizar la nube de puntos si se solicita
    if normalize:
        point_cloud = process_numpy_point_cloud(point_cloud)
    prepare_point_cloud(point_cloud, colors)
    return point_cloud, colors




def point_cloud_correction(point_cloud: np.array, model: any) -> np.array:
    """
    Aplica correcciones a la nube de puntos utilizando un modelo preentrenado de regresion lineal o una función de transformación específica.
    
    :param point_cloud: Nube de puntos 3D como un array de numpy, donde cada punto es representado por sus coordenadas (x, y, z).
    :param model: Modelo o función preentrenado/a que se utilizará para aplicar correcciones a la nube de puntos.
    :return: Nube de puntos 3D corregida como un array de numpy..
    """
    # Ejemplo de corrección simple: asumir que el modelo es una función que recibe y retorna un np.array
    corrected_point_cloud = pcGen.point_cloud_correction(point_cloud, model)

    return corrected_point_cloud

def generate_combined_filtered_point_cloud(img_left: np.array, img_right: np.array, config: dict, method: str, use_roi: bool, use_max_disparity: bool, normalize: bool = True):
    """
    Genera una nube de puntos 3D filtrada combinada a partir de un par de imágenes estéreo utilizando el método especificado. 
    Esta función está diseñada para trabajar con toda la nube de puntos y aplicar filtros para detectar y combinar todas las personas u objetos de interés en una sola nube de puntos.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_roi: Booleano que indica si se debe aplicar una Región de Interés (ROI) durante el procesamiento.
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
    :param normalize: Booleano que indica si se debe normalizar la nube de puntos a una escala de unidad estándar.
    :return: Una nube de puntos 3D filtrada que combina todas las personas u objetos detectados, junto con los colores correspondientes.
    """
    # Calcula el mapa de disparidad utilizando la función adecuada
    disparity_map = compute_disparity(img_left, img_right, config, method)
    
    # Acceder a parámetros relevantes desde la configuración
    Q = np.array(config['camera_params']['Q_matrix'])
    fx = config['camera_params']['fx']
    fy = config['camera_params']['fy']
    cx1 = config['camera_params']['cx1']
    cx2 = config['camera_params']['cx2']
    cy = config['camera_params']['cy']
    baseline = config['camera_params']['baseline']

    # Generar nube de puntos 3D filtrada combinada
    if method == 'SGBM':
        point_cloud, colors, eps, min_samples = pcGen.generate_all_filtered_point_cloud(
            img_left, disparity_map, Q, use_roi=use_roi, use_max_disparity=use_max_disparity, 
            camera_type="matlab"
        )
    else:
        point_cloud, colors, eps, min_samples = pcGen_ML.generate_all_filtered_point_cloud(
            img_left, disparity_map, fx, fy, cx1, cx2, cy, baseline, use_roi=use_roi, use_max_disparity=use_max_disparity,
            camera_type="matlab"
        )

    # Normalizar la nube de puntos si se solicita
    if normalize:
        point_cloud = process_numpy_point_cloud(point_cloud)
    prepare_point_cloud(point_cloud, colors)
    return point_cloud, colors


def generate_individual_filtered_point_clouds(img_left: np.array, img_right: np.array, config: dict, method: str, use_roi: bool, use_max_disparity: bool, normalize: bool = True):
    """
    Genera y retorna listas separadas de nubes de puntos, colores y keypoints 3D para cada objeto detectado individualmente, utilizando un método específico de disparidad y configuraciones de filtrado avanzadas. Los keypoints se estructuran según YOLOv8.

    Args:
        img_left (np.array): Imagen del lado izquierdo como array de numpy.
        img_right (np.array): Imagen del lado derecho como array de numpy.
        config (dict): Diccionario de configuración para un perfil específico.
        method (str): Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
        use_roi (bool): Indica si aplicar una Región de Interés (ROI) durante el procesamiento.
        use_max_disparity (bool): Indica si utilizar la disparidad máxima para optimizar la nube de puntos.
        normalize (bool): Indica si normalizar la nube de puntos a una escala de unidad estándar.

    Returns:
        Tuple of lists: Contiene listas de nubes de puntos, colores y keypoints 3D, cada una correspondiente a un objeto detectado individualmente. Los keypoints se estructuran en la forma [[[x1, y1, z1], [x2, y2, z2], ...], [...]] donde cada lista interna representa los keypoints de una persona.
    """
    # Generar el mapa de disparidad utilizando la función adecuada
    disparity_map = compute_disparity(img_left, img_right, config, method)

    # Acceder a parámetros relevantes desde la configuración
    Q = np.array(config['camera_params']['Q_matrix'])
    fx = config['camera_params']['fx']
    fy = config['camera_params']['fy']
    cx1 = config['camera_params']['cx1']
    cx2 = config['camera_params']['cx2']
    cy = config['camera_params']['cy']
    baseline = config['camera_params']['baseline']

    # Generar nubes de puntos filtradas para cada objeto detectado
    if method == 'SGBM':
        point_cloud_list, color_list, eps, min_samples, keypoints3d_list = pcGen.generate_filtered_point_cloud(
            img_left, disparity_map, Q, "matlab", use_roi, use_max_disparity
        )
    else:
        point_cloud_list, color_list, eps, min_samples, keypoints3d_list = pcGen_ML.generate_filtered_point_cloud(
            img_left, disparity_map, fx, fy, cx1, cx2, cy, baseline, "matlab", use_roi, use_max_disparity
        )

    # Normalizar las nubes de puntos si se solicita
    if normalize:
        normalized_point_cloud_list = [process_numpy_point_cloud(cloud) for cloud in point_cloud_list]
        normalized_keypoints_list = [process_numpy_point_cloud(kps) for kps in keypoints3d_list]
        
        prepare_individual_point_clouds(normalized_point_cloud_list, color_list, normalized_keypoints_list)
        return normalized_point_cloud_list, color_list, normalized_keypoints_list
    else:
        prepare_individual_point_clouds(point_cloud_list, color_list, keypoints3d_list)
        return point_cloud_list, color_list, keypoints3d_list



def compute_centroid(points, k=5, threshold_factor=1.0):
    if len(points) < k + 1:
        raise ValueError("La nube de puntos es demasiado pequeña para el valor de k.")

    # Construir un KDTree para la búsqueda de vecinos
    tree = KDTree(points)
    noise_mask = np.zeros(points.shape[0], dtype=bool)

    # Calcular la distancia euclidiana promedio entre todos los puntos
    distances, _ = tree.query(points, k=k+1)
    avg_distance = np.mean(distances[:, 1:])  # Omitir la distancia al propio punto
    threshold = threshold_factor * avg_distance

    for i, point in enumerate(points):
        # Encontrar los k vecinos más cercanos
        distances, indices = tree.query(point, k=k+1)
        # Omitir el propio punto
        distances = distances[1:]
        indices = indices[1:]

        # Calcular la profundidad media
        mean_depth = np.mean(distances)

        # Marcar el punto como ruido si su profundidad excede el umbral
        if np.any(np.abs(distances - mean_depth) > threshold):
            noise_mask[i] = True

    # Filtrar los puntos no ruidosos
    filtered_points = points[~noise_mask]

    if len(filtered_points) == 0:
        raise ValueError("Todos los puntos fueron considerados como ruido.")

    # Calcular el centroide de los puntos no ruidosos
    centroid = np.mean(filtered_points, axis=0)

    return centroid

def compute_centroids(points, k=5, threshold_factor=1.0, eps_factor=2, min_samples=10):
    if len(points) < k + 1:
        raise ValueError("La nube de puntos es demasiado pequeña para el valor de k.")

    # Construir un KDTree para la búsqueda de vecinos
    tree = KDTree(points)

    # Calcular la distancia euclidiana promedio entre todos los puntos
    distances, _ = tree.query(points, k=k+1)
    avg_distance = np.mean(distances[:, 1:])  # Omitir la distancia al propio punto
    threshold = threshold_factor * avg_distance
    eps = eps_factor * avg_distance

    noise_mask = np.zeros(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        # Encontrar los k vecinos más cercanos
        distances, indices = tree.query(point, k=k+1)
        # Omitir el propio punto
        distances = distances[1:]
        indices = indices[1:]

        # Calcular la profundidad media
        mean_depth = np.mean(distances)

        # Marcar el punto como ruido si su profundidad excede el umbral
        if np.any(np.abs(distances - mean_depth) > threshold):
            noise_mask[i] = True

    # Filtrar los puntos no ruidosos
    filtered_points = points[~noise_mask]

    if len(filtered_points) == 0:
        raise ValueError("Todos los puntos fueron considerados como ruido.")

    # Usar DBSCAN para detectar clusters
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_points)
    labels = clustering.labels_

    centroids = []
    for label in set(labels):
        if label == -1:  # Ignorar el ruido detectado por DBSCAN
            continue
        cluster_points = filtered_points[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    return centroids


def estimate_height_from_point_cloud(point_cloud: np.array, k: int = 5, threshold_factor: float = 1.0, m_initial: float = 50.0):
    """
    Estima la altura de una persona a partir de una nube de puntos y calcula el centroide de los keypoints.

    Args:
        point_cloud (np.array): Nube de puntos 3D representada como un array numpy de forma (N, 3).
        k (int): Número de vecinos más cercanos para calcular el centroide.
        threshold_factor (float): Factor de umbral para eliminar el ruido en el cálculo del centroide.
        m_initial (float): Rango inicial para filtrar los puntos alrededor del centroide.

    Returns:
        Tuple[float, np.array]: Altura estimada de la persona y el centroide calculado.
    """
    try:
        # Calcular el centroide de la nube de puntos
        centroid = compute_centroid(point_cloud, k=k, threshold_factor=threshold_factor)

        # Filtrar los puntos de la nube en un rango óptimo basado en el centroide
        filtered_points = filter_points_by_optimal_range(point_cloud, centroid, m_initial)

        # Obtener los límites mínimos y máximos en Y (altura)
        y_min, y_max = get_Y_bounds(filtered_points)

        if y_min is not None and y_max is not None:
            # Calcular la altura como la diferencia entre Y_max y Y_min
            height = abs(y_max - y_min)
            print(f"Para el centroide con z = {centroid[2]}, el rango de Y es: Y_min = {y_min}, Y_max = {y_max}")
            print(f"La altura de la persona es de {height}\n")
            return height, centroid
        else:
            print("No se encontraron puntos en el rango óptimo para este centroide.")
            return None, centroid

    except ValueError as ve:
        print(f"Error al calcular el centroide: {ve}")
        return None, None

