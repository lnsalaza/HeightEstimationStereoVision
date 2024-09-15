import sys
import os

# Agregar la ruta raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dense_point_cloud.point_cloud import generate_dense_point_cloud, generate_combined_filtered_point_cloud, generate_individual_filtered_point_clouds, generate_filtered_point_cloud_with_features, estimate_height_from_point_cloud, rectify_images
from dense_point_cloud.util import convert_point_cloud_format
from typing import Optional
from api_util.profile_management import *

class Orchestrator:
    def __init__(self, img_left, img_right, initial_requirement: str = "dense", profile_name: str = "default_profile", method: str = "SGBM", normalize: bool = True, use_max_disparity: bool = True):
        """
        Inicializa el Orchestrator con las imágenes, requerimiento, perfil, método, y otros parámetros.
        
        Args:
            img_left: Imagen del lado izquierdo en formato array.
            img_right: Imagen del lado derecho en formato array.
            initial_requirement: El requerimiento inicial ('dense', 'nodense', 'features', 'height').
            profile_name: El nombre del perfil de calibración a utilizar.
            method: El método de disparidad a utilizar (por defecto 'SGBM').
            normalize: Si se normalizan las nubes de puntos (por defecto True).
            use_max_disparity: Si se utiliza la disparidad máxima (por defecto True).
        """
        self.img_left = img_left
        self.img_right = img_right
        self.requirement = initial_requirement
        self.profile_name = profile_name  # Se puede cambiar el perfil en tiempo real
        self.method = method  # Método de disparidad (e.g., 'SGBM', 'RAFT', 'SELECTIVE')
        self.normalize = normalize  # Controla si se normaliza la nube de puntos
        self.use_max_disparity = use_max_disparity  # Controla si se utiliza la disparidad máxima

    def set_images(self, img_left, img_right):
        """
        Cambia las imágenes de entrada para los módulos.
        """
        self.img_left = img_left
        self.img_right = img_right

    def set_requirement(self, requirement: str):
        """
        Cambia el requerimiento de operación.
        """
        self.requirement = requirement

    def set_profile(self, profile_name: str):
        """
        Cambia el perfil de calibración.
        """
        self.profile_name = profile_name

    def set_method_params(self, method: str, normalize: bool, use_max_disparity: bool):
        """
        Cambia los parámetros de disparidad y normalización.
        """
        self.method = method
        self.normalize = normalize
        self.use_max_disparity = use_max_disparity

    def execute(self):
        """
        Ejecuta el módulo correspondiente basado en el requerimiento actual.
        """
        # Leer las imágenes
        left_image = self.img_left
        right_image = self.img_right

        # Rectificar imágenes
        profile = load_profile(self.profile_name)
        if not profile:
            raise FileNotFoundError(f"Perfil {self.profile_name} no encontrado.")
        
        left_image_rect, right_image_rect = rectify_images(left_image, right_image, self.profile_name)

        if self.requirement == "dense":
            point_cloud, colors = generate_dense_point_cloud(left_image_rect, right_image_rect, profile, self.method, self.use_max_disparity, self.normalize)
            return point_cloud, colors
        elif self.requirement == "nodense":
            point_cloud, colors = generate_combined_filtered_point_cloud(left_image_rect, right_image_rect, profile, self.method, False, self.use_max_disparity, self.normalize)
            return point_cloud, colors
        elif self.requirement == "features":
            point_clouds, colors, keypoints, features = generate_filtered_point_cloud_with_features(
                left_image_rect, right_image_rect, profile, self.method, False, self.use_max_disparity, self.normalize
            )
            return features
        elif self.requirement == "height":
            point_clouds_list, colors_list, keypoints3d_list = generate_individual_filtered_point_clouds(
                left_image_rect, right_image_rect, profile, self.method, False, self.use_max_disparity, self.normalize
            )
            results = []
            for point_cloud in point_clouds_list:
                height, centroid = estimate_height_from_point_cloud(point_cloud)
                results.append({"height": height, "centroid": centroid})
            return results
        else:
            raise ValueError(f"Requerimiento desconocido: {self.requirement}")



# Ejemplo de uso:
if __name__ == "__main__":
    # Leer imágenes usando OpenCV o cualquier librería que prefieras (esto es solo un ejemplo)
    import cv2
    img_left_array = cv2.imread("left_image.png")
    img_right_array = cv2.imread("right_image.png")

    orchestrator = Orchestrator(img_left_array, img_right_array, initial_requirement="dense")
    
    # Ejecutar una iteración inicial
    result = orchestrator.execute()
    print("Resultado inicial:", result)

    # Cambiar las imágenes y el requerimiento en tiempo real
    img_left_array_new = cv2.imread("new_left_image.png")
    img_right_array_new = cv2.imread("new_right_image.png")
    
    orchestrator.set_images(img_left_array_new, img_right_array_new)
    orchestrator.set_requirement("features")
    result = orchestrator.execute()
    print("Resultado con características:", result)