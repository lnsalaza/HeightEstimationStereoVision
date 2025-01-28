import json
import cv2

import numpy as np
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go

import calibration.calibration as cb
import api_util.profile_management as pm
from dense_point_cloud.point_cloud import * 
from dense_point_cloud.util import convert_point_cloud_format, convert_individual_point_clouds_format
from testing_util import convert_to_gray, test_convert_video

import numpy as np
class DotDict:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def generate_pointcloud(color_image, depth_image, fx, fy, cx, cy, depth_scale=1.0):
    """
    Genera una nube de puntos (points_3d, colors_3d) a partir
    de una imagen de color y un mapa de profundidad alineados.

    Parámetros:
    -----------
    color_image : np.ndarray, shape (H, W, 3)
        Imagen de color (BGR o RGB, según tu conveniencia).
    depth_image : np.ndarray, shape (H, W)
        Mapa de profundidad alineado con color_image.
        Cada valor depth_image[v, u] es la profundidad en alguna unidad
        (por ejemplo, mm). Se puede ajustar con depth_scale.
    fx, fy : float
        Focales en píxeles (ej. intrinsics.fx, intrinsics.fy).
    cx, cy : float
        Punto principal (center_x, center_y) en píxeles.
    depth_scale : float, opcional
        Factor para convertir la profundidad a la unidad deseada.
        Ejemplo: si depth_image está en mm y quieres metros => 0.001.
                 si ya está en la unidad que deseas => 1.0.

    Retorna:
    --------
    points_3d : np.ndarray, shape (N, 3)
        Coordenadas 3D de los puntos válidos.
    colors_3d : np.ndarray, shape (N, 3)
        Colores (B, G, R o R, G, B) de cada punto, alineados con points_3d.
    """

    # Asegurarnos de que color_image y depth_image tengan la misma forma HxW
    assert color_image.shape[:2] == depth_image.shape[:2], \
        "La imagen de color y el mapa de profundidad deben tener la misma resolución (HxW)."

    H, W = depth_image.shape

    # Generamos una grilla de coordenadas (u, v)
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten para manejar (u, v) como vectores 1D
    u_flat = u_coords.flatten().astype(np.float32)
    v_flat = v_coords.flatten().astype(np.float32)

    # Extraemos la profundidad y le aplicamos el factor de escala
    z_flat = depth_image.flatten().astype(np.float32) * depth_scale

    # Filtramos donde la profundidad sea 0 o negativa (sin datos)
    valid_mask = (z_flat > 0)

    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]
    z_valid = z_flat[valid_mask]

    # Ecuación pinhole inversa:
    #   X = (u - cx) * Z / fx
    #   Y = (v - cy) * Z / fy
    #   Z = Z
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy

    # Apilamos en (N, 3)
    points_3d = np.vstack((x_valid, y_valid, z_valid)).T  # shape (N, 3)

    # Extraemos colores correspondientes de color_image
    color_flat = color_image.reshape(-1, 3)
    colors_3d = color_flat[valid_mask]  # (N, 3)

    return points_3d, colors_3d


def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config
def xy_to_xyz(xy_coords, depth_image, depth_scale, intrinsics, to_unit="m", is_video=False):
    """
    Convertir coordenadas 2D (x, y) y una imagen de profundidad a coordenadas 3D (X, Y, Z).

    Args:
        xy_coords (list or np.ndarray): Lista o array de coordenadas (x, y) en píxeles.
        depth_image (np.ndarray): Imagen de profundidad (en píxeles).
        depth_scale (float): Escala de profundidad (en metros por unidad).
        intrinsics (rs.intrinsics): Parámetros intrínsecos de la cámara.

    Returns:
        np.ndarray: Coordenadas 3D (X, Y, Z) correspondientes a las coordenadas (x, y).
    """
    if to_unit == "m":
        factor = 1  # De metros a milímetros
    elif to_unit == "mm":
        factor = 1000  # De metros a milímetros
    elif to_unit == "cm":
        factor = 100  # De metros a centímetros
    else:
        raise ValueError("Unidad no válida. Usa 'mm' para milímetros o 'cm' para centímetros.")

    # Obtener las intrínsecas de la cámara
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    # Lista para almacenar las coordenadas 3D
    points_3d = []

    # Iterar sobre las coordenadas (x, y)
    for x, y in xy_coords:
        # dentro de los límites de la imagen
        if 0 < y < depth_image.shape[0] and 0 < x < depth_image.shape[1]:
            # Obtener la profundidad en la posición (x, y)
            if is_video:
                Z = depth_image[int(y), int(x)][0] * depth_scale
            else:
                Z = depth_image[int(y), int(x)] * depth_scale 

            # Calcular las coordenadas 3D
            X = (x - cx) * Z / fx
            Y = (y - cy) * Z / fy

            points_3d.append([X * factor, Y * factor, Z * factor])
        else:
            points_3d.append([0, 0, 0])

    # Convertir la lista a un array de NumPy
    return np.array(points_3d)

def generate_pointcloud_intrinsics(color_image, depth_image, intrinsics, to_unit="m", is_video=False):
    """
    Genera una nube de puntos densa (points_3d, colors_3d) usando la función xy_to_xyz
    y los intrínsecos contenidos en 'intrinsics'.

    Parámetros:
    -----------
    color_image : np.ndarray, shape (H, W, 3)
        Imagen de color (BGR o RGB).
    depth_image : np.ndarray, shape (H, W) o (H, W, 1)
        Mapa de profundidad (cada píxel = valor de profundidad).
    intrinsics : DotDict
        Debe contener al menos: fx, fy, ppx, ppy y depth_scale.
    to_unit : str
        "m", "cm" o "mm". Define a qué unidad convertir la salida.
    is_video : bool
        Si el mapa de profundidad proviene de un video donde cada píxel
        puede ser un vector o tener cierta forma distinta. Generalmente False.

    Retorna:
    --------
    points_3d : np.ndarray, shape (N, 3)
        Coordenadas 3D de cada píxel con profundidad válida.
    colors_3d : np.ndarray, shape (N, 3)
        Colores (B, G, R o R, G, B) de cada punto, alineados con points_3d.
    """
    # Verificar que color y profundidad tengan la misma HxW
    Hc, Wc = color_image.shape[:2]
    Hd, Wd = depth_image.shape[:2]
    if (Hc != Hd) or (Wc != Wd):
        raise ValueError("La imagen de color y la de profundidad deben tener la misma resolución HxW.")

    # Convertir la imagen de color a RGB si la tienes en BGR
    # (Dependiendo de cómo la hayas cargado en cv2, vendrá en BGR. 
    #  Si prefieres quedarte en BGR, también está bien, pero sé consistente.)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # 1) Generar todas las coordenadas (x, y) de la imagen
    x_coords, y_coords = np.meshgrid(np.arange(Wc), np.arange(Hc))
    # Apilar en un array de forma (N, 2)
    xy_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)

    # 2) Llamar a la función xy_to_xyz con estas coordenadas
    points_3d_all = xy_to_xyz(
        xy_coords=xy_coords,
        depth_image=depth_image,
        depth_scale=intrinsics.depth_scale,
        intrinsics=intrinsics,
        to_unit=to_unit,
        is_video=is_video
    )
    # points_3d_all es un array (N, 3)

    # 3) Extraer sólo los puntos que tienen Z > 0 para evitar basura en la nube
    #    (Muchos mapas de profundidad ponen 0 para "sin datos").
    valid_mask = points_3d_all[:, 2] > 0
    points_3d = points_3d_all[valid_mask]

    # 4) Emparejar colores: "aplanamos" la imagen de color y aplicamos la misma máscara
    colors_flat = color_image_rgb.reshape(-1, 3)
    colors_3d = colors_flat[valid_mask]

    return points_3d, colors_3d

if __name__ == "__main__":
    import json
    import cv2
    import open3d as o3d
    import numpy as np

    # Asegúrate de que tu clase DotDict esté bien definida
    class DotDict:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Creamos el objeto con intrínsecos
    depth_intrinsics = DotDict(
        depth_scale=0.0010000000474974513,
        fx=631.189453125,
        fy=631.189453125,
        ppx=647.0123901367188,
        ppy=362.94287109375
    )

    # Cargamos la imagen de color y la de profundidad
    intel_image = cv2.imread("../PC_2025/pc/Intel/depth/300/25_01_11_15_30_232691_original.jpg", cv2.IMREAD_COLOR)
    disparity_map = cv2.imread("../PC_2025/pc/Intel/depth/300/25_01_11_15_30_232691_depth.png", cv2.IMREAD_UNCHANGED)
    
    if intel_image is None or disparity_map is None:
        raise FileNotFoundError("Verifica las rutas de las imágenes.")

    # Generamos la nube de puntos usando la función adaptada
    points_3d, colors_3d = generate_pointcloud_intrinsics(
        color_image=intel_image,
        depth_image=disparity_map,
        intrinsics=depth_intrinsics,  # <-- usamos el DotDict
        to_unit="mm",                  # convertimos a metros (puedes usar "mm" o "cm" si quieres)
        is_video=False                # asume que NO es un frame de un video con canales extra
    )

    print("Se obtuvieron", points_3d.shape[0], "puntos en la nube.")

    # Crear un objeto PointCloud de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
    # Normaliza colores (0-1) si están en (0-255)
    pcd.colors = o3d.utility.Vector3dVector((colors_3d / 255.0).astype(np.float64))

    # Visualizar la nube de puntos en una ventana de Open3D
    o3d.visualization.draw_geometries([pcd], window_name='Nube de puntos')


    # 4) Visualizar con matplotlib
    #    Si la nube es muy grande, submuestreamos para no saturar la gráfica
    #    Ej.: tomamos solo 1 de cada 200 puntos
    # sample_step = 200
    # points_sample = points_3d[::sample_step]
    # colors_sample = colors_3d[::sample_step] / 255.0  # Normalizamos a [0..1]

    # # Crear la figura y el eje 3D
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Desempaquetar los ejes X, Y, Z
    # X = points_sample[:, 0]
    # Y = points_sample[:, 1]
    # Z = points_sample[:, 2]

    # # Hacemos un scatter 3D
    # sc = ax.scatter(
    #     X, Y, Z,
    #     c=colors_sample,  # array Nx3 en [0..1]
    #     marker='.',
    #     s=1  # tamaño del punto
    # )

    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.set_title("Nube de puntos (subsample)")

    # plt.show()
