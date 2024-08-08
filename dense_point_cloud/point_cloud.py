import json
import os
import cv2
import csv
import string
import joblib
import numpy as np
import dense_point_cloud.pc_generation as pcGen
import dense_point_cloud.pc_generation_ML as pcGen_ML
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from dense_point_cloud.Selective_IGEV.bridge_selective import get_SELECTIVE_disparity_map
from dense_point_cloud.RAFTStereo.bridge_raft import get_RAFT_disparity_map
from calibration.rectification import load_stereo_maps 
fx1 = 1429.4995220185822
fy1 = 1430.4111785502332

fx2 = 1433.6695087748499
fy2 = 1434.7285140471024

fx = (fx1 + fx2) / 2
fy = (fy1 + fy2) / 2

cx1 = 929.8227256572083
cy1 = 506.4722541384677

cx2 = 936.8035788332203
cy2 = 520.1168815891416

cy = (cy1 + cy2) / 2
baseline = 32.95550620237698 

"""
disparity_maps = get_RAFT_disparity_map(
        img_left, img_right, 
        restore_ckpt="models/raftstereo-middlebury.pth",
    )
"""
 # Asegúrate de tener esta función

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



import numpy as np

def generate_dense_point_cloud(img_left: np.array, img_right: np.array, config: dict, method: str, use_max_disparity: bool):
    """
    Genera una nube de puntos 3D densa a partir de un par de imágenes estéreo utilizando el método especificado.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
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

def generate_combined_filtered_point_cloud(img_left: np.array, img_right: np.array, config: dict, method: str, use_roi: bool, use_max_disparity: bool):
    """
    Genera una nube de puntos 3D filtrada combinada a partir de un par de imágenes estéreo utilizando el método especificado. 
    Esta función está diseñada para trabajar con toda la nube de puntos y aplicar filtros para detectar y combinar todas las personas u objetos de interés en una sola nube de puntos.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_roi: Booleano que indica si se debe aplicar una Región de Interés (ROI) durante el procesamiento.
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
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
            camera_type="matlab" #QUEMADO SE DEBE ACTUALIZAR 
        )
    else:
        point_cloud, colors, eps, min_samples = pcGen_ML.generate_all_filtered_point_cloud(
            img_left, disparity_map, fx, fy, cx1, cx2, cy, baseline, use_roi=use_roi, use_max_disparity=use_max_disparity,
            camera_type="matlab" #QUEMADO SE DEBE ACTUALIZAR 
        )

    return point_cloud, colors

def generate_individual_filtered_point_clouds(img_left: np.array, img_right: np.array, config: dict, method: str, use_roi: bool, use_max_disparity: bool):
    """
    Genera y retorna listas separadas de nubes de puntos y colores para cada objeto detectado individualmente, utilizando un método específico de disparidad y configuraciones de filtrado avanzadas.

    :param img_left: Imagen del lado izquierdo como array de numpy.
    :param img_right: Imagen del lado derecho como array de numpy.
    :param config: Diccionario de configuración para un perfil específico.
    :param method: Método de disparidad a utilizar (e.g., 'SGBM', 'RAFT', 'SELECTIVE').
    :param use_roi: Booleano que indica si se debe aplicar una Región de Interés (ROI) durante el procesamiento.
    :param use_max_disparity: Booleano que indica si se debe utilizar la disparidad máxima para optimizar la nube de puntos.
    :return: Listas de nubes de puntos y colores, cada una correspondiente a un objeto detectado individualmente.
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
        point_cloud_list, color_list, eps, min_samples = pcGen.generate_filtered_point_cloud(
            img_left, disparity_map, Q, "matlab", use_roi, use_max_disparity,
            
        )
    else:
        point_cloud_list, color_list, eps, min_samples = pcGen_ML.generate_filtered_point_cloud(
            img_left, disparity_map, fx, fy, cx1, cx2, cy, baseline,"matlab", use_roi, use_max_disparity,
           
        )

    return point_cloud_list, color_list

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
######################### NEXT FUNCTION ARE JUST FOR TESTING PURPOSES #################################
def test_disparity_map(img_left, img_right, config, method):
    # Calcular el mapa de disparidad
    disparity_map = compute_disparity(img_left, img_right, config, method)


    # Visualizar el mapa de disparidad generado
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

def test_point_cloud(img_left, img_right, config, method, use_max_disparity):
    # Generar la nube de puntos 3D
    point_cloud, colors = generate_dense_point_cloud(img_left, img_right, config, method, use_max_disparity)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/densaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="3D Point Cloud", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 1  # Establecer el tamaño de los puntos

    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)

    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Crea bounding boxes de los puntos en la nube
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    viewer.add_geometry(aabb)

    
    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    if not use_roi:
        opt.point_size = 5  # Establecer el tamaño de los puntos
    else:
        opt.point_size = 1
    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)

    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Calcular los centroides de los clusters/personas
    centroids = compute_centroids(point_cloud)

    print(f"CENTROIDES: {centroids}")
    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud with Centroids", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Añadir las esferas de los centroides a la ventana de visualización
    for centroid in centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)  # Ajusta el radio según sea necesario
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo
        viewer.add_geometry(sphere)

    # Crear bounding boxes de los puntos en la nube
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    viewer.add_geometry(aabb)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    if not use_roi:
        opt.point_size = 5  # Establecer el tamaño de los puntos
    else:
        opt.point_size = 1
    
    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity)
    
    for i, (point_cloud, colors) in enumerate(zip(point_cloud_list, color_list)):
        # Convertir los datos de la nube de puntos y colores a formato Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear una ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos a la ventana de visualización
        viewer.add_geometry(pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Establecer el tamaño de los puntos
        else:
            opt.point_size = 1
        # Ejecutar la visualización
        
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()


def test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity)
    
    for i, (point_cloud, colors) in enumerate(zip(point_cloud_list, color_list)):
        # Calcular el centroide omitiendo puntos ruidosos
        centroid = compute_centroid(point_cloud)
        print(f"CENTROIDE:  {centroid}")

        # Convertir los datos de la nube de puntos y colores a formato Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear una nube de puntos para el centroide (VARIANTE 1)
        # centroid_pcd = o3d.geometry.PointCloud()
        # centroid_pcd.points = o3d.utility.Vector3dVector(np.array([centroid]))
        # centroid_pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))  # Rojo

        # Crear una esfera para el centroide (VARIANTE 2)
        centroid_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=5)  # Ajusta el radio según sea necesario
        centroid_pcd.translate(centroid)
        centroid_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo

        # Crear una ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos y el centroide a la ventana de visualización
        viewer.add_geometry(pcd)
        viewer.add_geometry(centroid_pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Establecer el tamaño de los puntos
        else:
            opt.point_size = 1
        
        # Ejecutar la visualización
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()




def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Cargar las imágenes como arrays
    img_left = cv2.imread("../images/calibration_results/matlab_1/flexometer/250 y 600/14_13_13_13_05_2024_IMG_LEFT.jpg")
    img_right = cv2.imread("../images/calibration_results/matlab_1/flexometer/250 y 600/14_13_13_13_05_2024_IMG_RIGHT.jpg")
    
    if img_left is None or img_right is None:
        raise FileNotFoundError("Una o ambas imágenes no pudieron ser cargadas. Verifique las rutas.")

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)


    # Cargar configuración desde el archivo JSON
    config = load_config("../profiles/profile1.json")
    
    # Asumiendo que queremos usar el método SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
    method = 'SELECTIVE'

    # #TEST MAPA DISPARIDAD
    # test_disparity_map(img_left, img_right, config, method)

    # #TEST NUBE DE PUNTOS DENSA
    # test_point_cloud(img_left, img_right, config, method, use_max_disparity=False)


    # #TEST NUBE DE PUNTOS NO DENSA TOTAL
    test_filtered_point_cloud(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA TOTAL
    # test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)



    # #TEST NUBE DE PUNTOS NO DENSA INDIVIDUAL
    # test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA INDIVIDUAL
    # test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

