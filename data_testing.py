import json
import string
import cv2
import csv
import numpy as np
import pandas as pd
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from dense_point_cloud.point_cloud import * 
from dense_point_cloud.util import convert_point_cloud_format, convert_individual_point_clouds_format
from testing_util import *



# data = []
# data_height = []
# camera_type = 'matlab_1'
# mask_type = 'keypoint'

# is_roi = (mask_type == "roi")

# method_used = "SGBM" #OPTIONS: "SGBM". 'WLS-SGBM', "RAFT", "SELECTIVE"
# apply_correction = False


# print(f"{method_used} ESTA SIENDO USADO")


# pairs = read_image_pairs_by_distance('../images/calibration_results/matlab_1/flexometer')
# alphabet = string.ascii_lowercase
# alturas = []


def save_dataset(data, base_filename):
    # Convertir la lista de datos a un DataFrame de pandas
    df = pd.DataFrame(data, columns=['situation', 'letter', 'person_index', 'depth', 'height'])
    
    # Crear el nombre del archivo completo agregando la extensión .csv
    filename = f"{base_filename}.csv"
    
    # Obtener el directorio desde el nombre del archivo
    directory = os.path.dirname(filename)
    
    # Verificar si el directorio existe, y si no, crearlo
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")
    
    # Guardar el DataFrame como CSV
    df.to_csv(filename, index=False)
    
    print(f"Dataset guardado correctamente en {filename}")

def save_dataset_path(data, base_filename):
    df = pd.DataFrame(data, columns=['situation_depth', 'situation_height', 'letter', 'person_index', 'depth', 'height'])
    filename = f"{base_filename}.csv"
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_csv(filename, index=False)
    print(f"Dataset guardado correctamente en {filename}")

def process_situation(situation, variations, configs, camera_type, method_used, use_roi, mask_type, normalize):
    data = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for variation, letter in zip(variations, alphabet):
        print(f"\nProcesando situación: {situation} | Variante {letter}")
        
        img_left = cv2.imread(variation[0])
        img_right = cv2.imread(variation[1])
        
        # Rectificar imágenes según perfil de calibración
        img_left, img_right = rectify_images(img_left, img_right, configs['profile_name'])
        
        # Generar y procesar nubes de puntos filtradas individuales
        point_cloud_list, colors_list, keypoints3d_list = generate_individual_filtered_point_clouds(
            img_left, img_right, configs, method_used, use_roi, use_max_disparity=True, normalize=normalize
        )
        
        # Calcular la profundidad y altura para cada conjunto de keypoints
        for person_index, (point_cloud, keypoints3d) in enumerate(zip(point_cloud_list, keypoints3d_list)):
            height, centroid = estimate_height_from_point_cloud(point_cloud, m_initial=100)
            depth = centroid[2]  # Asumiendo que el centroide Z representa la profundidad
            
            # Agregar a la lista de datos incluyendo el índice de la persona
            data.append([situation, letter, person_index + 1, depth, height])  # Se suma 1 para empezar en persona 1

    return data


def process_all_situations(pairs, configs, camera_type, method_used, is_roi, mask_type, normalize):
    all_data = []
    base_filename = f"./datasets/data/estable/NO-NORM-{method_used}_HEIGHT"
    for situation, variations in pairs.items():
        try:
            data = process_situation(situation, variations, configs, camera_type, method_used, is_roi, mask_type, normalize=normalize)
            all_data.extend(data)
        except Exception as e:
            print(f"Error procesando {situation}: {e}")

    save_dataset(all_data, base_filename)



#######################################################################################################################


def process_all_situations_path(root_folder, configs, camera_type, method_used, is_roi, mask_type, normalize):
    all_data = []
    base_filename = f"./datasets/data/estable/EXTRA/NORM-{method_used}_EXTRA"

    # Recorrer las carpetas de profundidad
    for depth_folder in os.listdir(root_folder):
        depth_path = os.path.join(root_folder, depth_folder)
        if os.path.isdir(depth_path):  # Asegurarse de que sea un directorio
            # Recorrer las subcarpetas de altura dentro de cada carpeta de profundidad
            for height_folder in os.listdir(depth_path):
                height_path = os.path.join(depth_path, height_folder)
                if os.path.isdir(height_path):  # Asegurarse de que sea un directorio
                    situation_depth = depth_folder
                    situation_height = height_folder
                    image_pairs = []

                    # Recoger pares de imágenes dentro de la carpeta de altura
                    for file in os.listdir(height_path):
                        if file.endswith('LEFT.jpg'):
                            right_file = file.replace('LEFT', 'RIGHT')
                            if right_file in os.listdir(height_path):
                                left_path = os.path.join(height_path, file)
                                right_path = os.path.join(height_path, right_file)
                                image_pairs.append((left_path, right_path))

                    # Procesar las imágenes recolectadas
                    if image_pairs:
                        data = process_situation_path(situation_depth, situation_height, image_pairs, configs, camera_type, method_used, is_roi, mask_type, normalize)
                        all_data.extend(data)

    save_dataset_path(all_data, base_filename)

def process_situation_path(situation_depth, situation_height, image_pairs, configs, camera_type, method_used, use_roi, mask_type, normalize):
    data = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    for index, (left_path, right_path) in enumerate(image_pairs):
        letter = alphabet[index % len(alphabet)]  # Asegurar no salirse del abecedario
        print(f"\nProcesando situación: {situation_depth}_{situation_height} | Variante {letter}")

        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)

        img_left, img_right = rectify_images(img_left, img_right, configs['profile_name'])

        point_cloud_list, colors_list, keypoints3d_list = generate_individual_filtered_point_clouds(
            img_left, img_right, configs, method_used, use_roi, use_max_disparity=True, normalize=normalize
        )

        for person_index, (point_cloud, keypoints3d) in enumerate(zip(point_cloud_list, keypoints3d_list)):
            height, centroid = estimate_height_from_point_cloud(point_cloud)
            depth = centroid[2]  # Asumiendo que el centroide Z representa la profundidad

            data.append([situation_depth, situation_height, letter, person_index + 1, depth, height])  # Se suma 1 para empezar en persona 1

    return data





if __name__ == "__main__":
    # Aquí puedes inicializar las variables necesarias, por ejemplo:

    configs = load_config("profiles/MATLAB.json")  # Tu configuración específica
    method_used = "SELECTIVE"  # O el método que uses, como 'WLS-SGBM', "RAFT", "SELECTIVE"
    is_roi = False  # O True, dependiendo de si usas ROI
    apply_correction = True  # O False, dependiendo si aplicas corrección
    model = None  # Define tu modelo de corrección si es necesario
    mask_type = "keypoints"  # Define el tipo de máscara
    
    # pairs = read_image_pairs_by_distance("../originals/precision height x depth") #Tu diccionario de situaciones y variaciones

    # # Llamar a la función principal
    # process_all_situations_path("../originals/precision height x depth", configs, method_used, method_used, is_roi, mask_type, apply_correction)


    