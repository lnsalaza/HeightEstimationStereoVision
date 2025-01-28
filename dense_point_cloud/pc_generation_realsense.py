import os
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

# Importa tu módulo de keypoint_extraction si lo necesitas:
import dense_point_cloud.keypoint_extraction as kp

##############################################################################
#                           Parámetros (opcionales)                          #
##############################################################################
sigma = 1.5   # Ejemplo para manejar si quieres filtrar la profundidad (opcional).
lmbda = 8000.0
class DotDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


##############################################################################
#                     Funciones de Utilidad (Guardar imágenes)               #
##############################################################################
def save_image(path, image, image_name, grayscale=False):
    """
    Guarda una imagen con un nombre incremental en la carpeta especificada.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    files = os.listdir(path)
    image_files = [f for f in files if f.startswith(image_name)]
    next_number = len(image_files) + 1
    new_image_filename = f'{image_name}_{next_number}.png'
    full_path = os.path.join(path, new_image_filename)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(full_path, image)


##############################################################################
#      Función principal de reproyección: depth_to_pointcloud (RealSense)    #
##############################################################################
def depth_to_pointcloud(
    depth_image,
    color_image,
    intrinsics,    # Debe contener fx, fy, ppx, ppy, depth_scale
    custom_mask=None,
    is_video=False,
    to_unit="mm"
):
    """
    Convierte un mapa de profundidad y su imagen de color correspondiente en una
    nube de puntos 3D (N, 3) con sus colores (N, 3). 
    Opcionalmente aplica una máscara booleana para filtrar regiones.

    Parámetros:
    -----------
    depth_image: np.ndarray (H, W) o (H, W, 1)
        Imagen de profundidad (16 bits o 8 bits, cargada con IMREAD_UNCHANGED).
    color_image: np.ndarray (H, W, 3)
        Imagen de color alineada con depth_image (BGR o RGB).
    intrinsics: DotDict u objeto con:
        - fx, fy, ppx, ppy: parámetros intrínsecos de la cámara
        - depth_scale: factor para convertir el valor del depth_image a metros (ej.: 0.001)
    custom_mask: np.ndarray (H, W), opcional
        Máscara booleana para descartar píxeles no deseados (ej.: segmentación).
    is_video: bool
        Si la profundidad proviene de un video con un formato especial (por ejemplo, 3 canales).
        Usar con precaución: por defecto es False.
    to_unit: str
        "m", "mm" o "cm". Unidad final de la nube de puntos.

    Retorna:
    --------
    out_points: np.ndarray (N, 3)
    out_colors: np.ndarray (N, 3)
        Nube de puntos filtrada + colores correspondientes.
    """
    # Asegúrate de que color y profundidad tengan misma resolución
    assert depth_image.shape[:2] == color_image.shape[:2], \
        "La imagen de color y la de profundidad deben tener la misma resolución HxW."

    # Convertir color a RGB si está en BGR (opcional, según tu conveniencia)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Extraer intrínsecos
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    depth_scale = intrinsics.depth_scale

    # Determinar factor de unidad
    if to_unit == "m":
        factor = 1.0
    elif to_unit == "mm":
        factor = 1000.0
    elif to_unit == "cm":
        factor = 100.0
    else:
        raise ValueError("Unidad inválida. Usa 'm', 'mm' o 'cm'.")

    H, W = depth_image.shape[:2]

    # Generar todas las coordenadas 2D (x, y)
    x_coords, y_coords = np.meshgrid(np.arange(W), np.arange(H))
    x_coords = x_coords.ravel()
    y_coords = y_coords.ravel()

    # Leer profundidad
    if is_video:
        # Caso especial si la profundidad viene en un formato [H, W, 3] o similar
        z_vals = depth_image.reshape(-1, 3)[:, 0].astype(np.float32) * depth_scale
    else:
        z_vals = depth_image.reshape(-1).astype(np.float32) * depth_scale

    # Crear máscara válida (Z > 0)
    valid_mask = z_vals > 0

    # Si se proporciona una máscara externa, combinarla
    if custom_mask is not None:
        # custom_mask debe ser bool
        mask_flat = custom_mask.ravel() > 0
        valid_mask &= mask_flat

    # Aplicar la máscara para quedarnos con los píxeles válidos
    x_valid = x_coords[valid_mask]
    y_valid = y_coords[valid_mask]
    z_valid = z_vals[valid_mask]

    # Reproyección pinhole inversa
    X = (x_valid - cx) * z_valid / fx
    Y = (y_valid - cy) * z_valid / fy
    Z = z_valid

    # Escala final (a mm/cm si se desea)
    X *= factor
    Y *= factor
    Z *= factor

    out_points = np.vstack((X, Y, Z)).T  # (N, 3)

    # Extraer colores
    color_flat = color_image_rgb.reshape(-1, 3)
    out_colors = color_flat[valid_mask]

    return out_points, out_colors


##############################################################################
#      DBSCAN, centroides y creación/guardado de nubes (idéntico a stereo)   #
##############################################################################
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
            print("z = ", str(centroid[2]))
        return np.array(centroids)

def create_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    return pcd

def save_point_cloud(point_cloud, colors, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pcd = create_point_cloud(point_cloud, colors)
    o3d.io.write_point_cloud(filename, pcd, print_progress=False)

def process_point_cloud(point_cloud, eps, min_samples, base_filename):
    labels = apply_dbscan(point_cloud, eps, min_samples)
    centroids = get_centroids(point_cloud, labels)

    if centroids is not None:
        centroid_colors = np.tile([[255, 0, 0]], (len(centroids), 1))  # Rojo
        centroid_filename = f"{base_filename}_centroids.ply"
        save_point_cloud(centroids, centroid_colors, centroid_filename)
        
    original_cloud_colors = np.ones_like(point_cloud) * [0, 0, 255]  # Azul
    original_filename = f"{base_filename}_original.ply"
    save_point_cloud(point_cloud, original_cloud_colors, original_filename)

    return centroids


##############################################################################
#     Reemplazamos "disparity_to_pointcloud" y funciones de stereo masks     #
##############################################################################
def generate_all_filtered_point_cloud(color_image, depth_image, intrinsics, camera_type, use_roi=True):
    """
    Genera la nube de puntos con un filtro de segmentación 'ROI' (o keypoints)
    para RealSense. Similar a generate_all_filtered_point_cloud en pc_generation.
    
    Retorna:
        point_cloud (N,3), colors (N,3), eps, min_samples
    """
    keypoints = []
    if use_roi:
        seg = kp.get_segmentation(color_image)
        mask = kp.apply_seg_mask(depth_image, seg)  # Ajusta si tu 'apply_seg_mask' requiere adaptaciones
        eps, min_samples = 2, 3500
    else:
        keypoints = kp.get_keypoints(color_image)
        mask = kp.apply_keypoints_mask(depth_image, keypoints) 
        eps = 50 if "matlab" in camera_type else 10
        min_samples = 6

    # Reproyectar usando depth_to_pointcloud
    point_cloud, colors = depth_to_pointcloud(
        depth_image=depth_image,
        color_image=color_image,
        intrinsics=intrinsics,
        custom_mask=mask,
        is_video=False,
        to_unit="m"
    )

    return point_cloud.astype(np.float64), colors.astype(np.float64), eps, min_samples


def generate_filtered_point_cloud(color_image, depth_image, intrinsics, camera_type, use_roi=True):
    """
    Versión que procesa múltiples máscaras (por ejemplo, cada segmento/ROI separado
    o cada keypoint), generando varias nubes. Similar a generate_filtered_point_cloud
    de la versión estéreo.
    
    Retorna:
        point_cloud_list, colors_list, eps, min_samples, keypoints3d
    """
    result_image_list = []
    point_cloud_list = []
    colors_list = []
    keypoints3d = []

    if use_roi:
        seg = kp.get_segmentation(color_image)
        for s in seg:
            s_list = [s]
            mask = kp.apply_seg_mask(depth_image, s_list)
            result_image_list.append(mask)
        eps, min_samples = 5, 1000

    else:
        kpts = kp.get_keypoints(color_image)
        for k in kpts:
            k_list = [k]
            mask = kp.apply_keypoints_mask(depth_image, k_list)
            result_image_list.append(mask)
        # Si quieres obtener keypoints en 3D, deberías iterar con xy_to_xyz
        # o volver a usar la lógica que corresponda
        keypoints3d = get_strutured_kepoints3d(kpts, depth_image, intrinsics)  # Ver abajo

        eps = 100 if "matlab" in camera_type else 10
        min_samples = 6

    # Reproyectar cada máscara
    for mask in result_image_list:
        pc, cols = depth_to_pointcloud(
            depth_image=depth_image,
            color_image=color_image,
            intrinsics=intrinsics,
            custom_mask=mask,
            is_video=False,
            to_unit="mm"
        )
        point_cloud_list.append(pc.astype(np.float64))
        colors_list.append(cols.astype(np.float64))

    return point_cloud_list, colors_list, eps, min_samples, keypoints3d


##############################################################################
#               Ejemplo de "keypoints 3D" (si lo deseas usar)                #
##############################################################################
def get_strutured_kepoints3d(keypoints, depth_image, intrinsics):
    """
    Similar a la versión estéreo, pero usando la función xy_to_xyz 
    (o la lógica de depth_to_pointcloud individual para cada keypoint).
    """
    # Asegúrate de que keypoints sea una lista de listas de (x,y).
    # Aquí simplemente se muestra un ejemplo de cómo podrías obtener
    # la coordenada 3D de cada keypoint usando la misma lógica de proyección.
    # Ajusta según tu definición real de "keypoints".
    estructura_con_coordenadas_3d = []

    H, W = depth_image.shape[:2]
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    depth_scale = intrinsics.depth_scale

    for persona in keypoints:
        persona_3d = []
        for (x, y) in persona:
            if 0 <= y < H and 0 <= x < W:
                z_val = depth_image[int(y), int(x)] * depth_scale
                if z_val > 0:
                    X = (x - cx) * z_val / fx
                    Y = (y - cy) * z_val / fy
                    Z = z_val
                    persona_3d.append([X, Y, Z])
                else:
                    persona_3d.append([0, 0, 0])
            else:
                persona_3d.append([0, 0, 0])
        estructura_con_coordenadas_3d.append(np.array(persona_3d, dtype=np.float64))
    
    return estructura_con_coordenadas_3d


##############################################################################
#         Corrección de nubes (regresión) y guardado, igual que stereo        #
##############################################################################
def point_cloud_correction(points, model):
    points = np.asarray(points)
    x = points[:, 0].reshape(-1,1)
    x_pred = model.predict(x)
    y = points[:, 1].reshape(-1,1)
    y_pred = model.predict(y)
    z = points[:, 2].reshape(-1,1)
    z_pred = model.predict(z)

    corrected_points = np.column_stack((x_pred, y_pred, z_pred))
    return corrected_points

def save_dense_point_cloud(point_cloud, colors, base_filename):
    if not os.path.exists(os.path.dirname(base_filename)):
        os.makedirs(os.path.dirname(base_filename))
    dense_filename = f"{base_filename}.ply"
    save_point_cloud(point_cloud, colors, dense_filename)
