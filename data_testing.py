import json
import string
import cv2
import csv
import numpy as np
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from dense_point_cloud.point_cloud import * 
from dense_point_cloud.util import convert_point_cloud_format, convert_individual_point_clouds_format
from testing_util import *


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



# data = []
# data_height = []
# camera_type = 'matlab_1'
# mask_type = 'keypoint'

# is_roi = (mask_type == "roi")

# method_used = "SGBM" #OPTIONS: "SGBM". "RAFT", "SELECTIVE"
# apply_correction = False


# print(f"{method_used} ESTA SIENDO USADO")


# pairs = read_image_pairs_by_distance('../images/calibration_results/matlab_1/flexometer')
# alphabet = string.ascii_lowercase
# alturas = []




def process_situation(situation, variations, configs, camera_type, method_used, use_roi, apply_correction, model, mask_type):
    data = []
    data_height = []
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for variation, letter in zip(variations, alphabet):
        print(f"\n\nProcesando situación: {situation} | Variante {letter}")
        
        img_left = cv2.imread(variation[0])
        img_right = cv2.imread(variation[1])
        # img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
        # img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        
        # Cargar y rectificar imágenes
        img_left, img_right = rectify_images(img_left, img_right, configs['profile_name'])
        

        # Generar nube de puntos densa
        point_cloud, colors = generate_dense_point_cloud(
            img_left, img_right, configs, method_used, use_max_disparity=True, normalize=False
        )
        
        base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}_{letter}"
        
        # Aplicar corrección a la nube de puntos si se solicita
        if apply_correction:
            point_cloud = point_cloud_correction(point_cloud, model)
            base_filename += "_corregido"
        
        # Guardar la nube de puntos densa
        pcGen.save_point_cloud(point_cloud, colors, base_filename)
        # pcGen.save_dense_point_cloud(point_cloud, colors, base_filename)
        
        # Generar y procesar nubes de puntos filtradas individuales
        point_cloud_list, colors_list, keypoints3d_list = generate_individual_filtered_point_clouds(
            img_left, img_right, configs[camera_type], method_used, use_roi, use_max_disparity=True
        )

        # Estimar alturas y guardar datos
        heights = []
        for idx, (pc, cl) in enumerate(zip(point_cloud_list, colors_list)):
            height, centroid = estimate_height_from_point_cloud(pc)
            if height is not None:
                heights.append(height)
                print(f"La altura de la persona {idx + 1} es de {height}\n")
            data.append({
                "situation": f"{situation}_{letter}",
                f"z_estimation_{idx + 1}": centroid[2] if centroid is not None else None,
                f"h_estimation_{idx + 1}": height
            })
        
        data_height.append(heights)

    return data, data_height

def save_dataset(data, method_used, camera_type, mask_type):
    if data:
        dataset_path = f"../datasets/data/{method_used}_z_estimation_{camera_type}_{mask_type}_train.csv"
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        max_z_count = max(len(row) - 1 for row in data)  # -1 porque 'situation' no es una columna z_estimation
        fieldnames = ["situation"] + [f"z_estimation_{i+1}" for i in range(max_z_count)] + [f"h_estimation_{i+1}" for i in range(max_z_count)]
        with open(dataset_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        print(f"Dataset guardado en {dataset_path}")

def process_all_situations(pairs, configs, camera_type, method_used, is_roi, apply_correction, model, mask_type):
    all_data = []
    
    for situation, variations in pairs.items():
        try:
            data, _ = process_situation(situation, variations, configs, camera_type, method_used, is_roi, apply_correction, model, mask_type)
            all_data.extend(data)
        except Exception as e:
            print(f"Error procesando {situation}: {e}")

    save_dataset(all_data, method_used, camera_type, mask_type)

if __name__ == "__main__":
    # Aquí puedes inicializar las variables necesarias, por ejemplo:

    configs = load_config("profiles/MATLAB.json")  # Tu configuración específica
    method_used = "SGBM"  # O el método que uses, como "RAFT", "SELECTIVE"
    is_roi = False  # O True, dependiendo de si usas ROI
    apply_correction = False  # O False, dependiendo si aplicas corrección
    model = None  # Define tu modelo de corrección si es necesario
    mask_type = "keypoints"  # Define el tipo de máscara
    

    pairs = read_image_pairs_by_distance("../HeightEstimationStereoVision/images/distances") #Tu diccionario de situaciones y variaciones




    # Llamar a la función principal
    process_all_situations(pairs, configs, method_used, method_used, is_roi, apply_correction, model, mask_type)