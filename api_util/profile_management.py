import os
import math
import json
from typing import Dict, Optional
import numpy as np
def save_profile(profile_data, profile_name, directory="profiles"):
    # Asegurarse de que el directorio existe; crearlo si no es así
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, f"{profile_name}.json")
    with open(file_path, 'w') as file:
        json.dump(profile_data, file)

    return file_path  # Opcional, devuelve la ruta del archivo para referencia

def generate_profile_data(calibration_data: Dict, profile_name: str, Q:any) -> Dict:
    """
    Genera datos de perfil estandarizados a partir de los parámetros de calibración.

    Args:
        calibration_data (Dict): Datos de calibración obtenidos del JSON de calibración.
        profile_name (str): Nombre del perfil.

    Returns:
        Dict: Datos del perfil formateados según la estructura requerida.
    """
    # Calcula el promedio geométrico de fx y fy para ambas cámaras
    fx = math.sqrt(calibration_data['flCamera1'][0] * calibration_data['flCamera2'][0])
    fy = math.sqrt(calibration_data['flCamera1'][1] * calibration_data['flCamera2'][1])
    
    # Calcula el promedio geométrico general para usar como f en Q_matrix
    f = math.sqrt(fx * fy)
    
    # Distancia entre las cámaras (baseline)
    Tx = calibration_data['stereoT'][0]
    
    # Genera la matriz Q utilizando los valores calculados
    # Q_matrix2 = [
    #     [1.0, 0.0, 0.0, -calibration_data['cameraMatrix1'][2][0]],
    #     [0.0, 1.0, 0.0, -calibration_data['cameraMatrix1'][2][1]],
    #     [0.0, 0.0, 0.0, f],  # Se usa el promedio geométrico de fx y fy aquí en caso de que no se tenga un fx, fy igual en las dos camaras. Idealmente deberia ser fx1=fy1=fx2=fy2
    #     [0.0, 0.0, -1.0/Tx, (calibration_data['cameraMatrix1'][2][0] - calibration_data['cameraMatrix2'][2][0]) / Tx]
    # ]
    
    baseline = np.linalg.norm(np.array(calibration_data['stereoT']))
    Q_matrix = Q.tolist()
    # Estructura del perfil a devolver
    return {
        "profile_name": profile_name,
        "resolution":calibration_data['imageSize'],
        "camera_params": {
            "fx": fx,
            "fy": fy,
            "cx1": calibration_data['cameraMatrix1'][2][0],
            "cx2": calibration_data['cameraMatrix2'][2][0],
            "cy": (calibration_data['cameraMatrix1'][2][1] + calibration_data['cameraMatrix2'][2][1]) / 2,
            "baseline": baseline,
            "Q_matrix": Q_matrix
        },
        "disparity_methods": {
            "SGBM": {
                "enabled": True,
                "name": "StereoSGBM",
                "params": {
                    "numDisparities": 68,
                    "blockSize": 7,
                    "minDisparity": 5,
                    "disp12MaxDiff": 33,
                    "uniquenessRatio": 10,
                    "speckleWindowSize": 50,
                    "speckleRange": 1,
                    "preFilterCap": 33,
                    "mode": "StereoSGBM_MODE_HH",
                    "wls_filter": False
                },
                "correction_model": ""
            },
            "WLS-SGBM": {
                "enabled": True,
                "name": "StereoSGBM-WLS",
                "params": {
                    "numDisparities": 68,
                    "blockSize": 7,
                    "minDisparity": 5,
                    "disp12MaxDiff": 33,
                    "uniquenessRatio": 10,
                    "speckleWindowSize": 50,
                    "speckleRange": 1,
                    "preFilterCap": 33,
                    "mode": "StereoSGBM_MODE_HH",
                    "wls_filter": True
                },
                "correction_model": ""
            },
            "RAFT": {
                "enabled": True,
                "name": "RAFT",
                "params": {
                    "restore_ckpt": "dense_point_cloud/RAFTStereo/models/raftstereo-middlebury.pth"
                },
                "correction_model": ""
            },
            "SELECTIVE": {
                "enabled": True,
                "name": "Selective",
                "params": {
                    "restore_ckpt": "dense_point_cloud/Selective_IGEV/pretrained_models/middlebury_train.pth"
                },
                "correction_model": ""
            }
        },
        "output_directory": "./output",
        "filename_template": "point_cloud_{timestamp}.ply"
    }
def list_profiles(directory="profiles"):
    profiles = []
    for file in os.listdir(directory):
        if file.endswith(".json"):
            profile_path = os.path.join(directory, file)
            profile_name = file[:-5]  # Remover la extensión .json para obtener el nombre
            profiles.append({"name": profile_name, "path": profile_path})
    return profiles

def delete_profile(profile_name: str, profile_dir="profiles", config_dir="config_files"):
    """
    Elimina un perfil y sus archivos relacionados basándose en el nombre del perfil.

    Args:
        profile_name (str): Nombre del perfil para eliminar.
        profile_dir (str): Directorio donde se guardan los archivos de perfil.
        config_dir (str): Directorio donde se guardan los archivos de configuración.
    
    Returns:
        bool: True si se eliminan los archivos correctamente, False si no.
    """
    profile_path = os.path.join(profile_dir, f"{profile_name}.json")
    config_path = os.path.join(config_dir, profile_name)
    
    try:
        # Eliminar el archivo de perfil JSON
        if os.path.exists(profile_path):
            os.remove(profile_path)
        else:
            return False
        
        # Eliminar todos los archivos relacionados en el directorio de configuración
        if os.path.isdir(config_path):
            for file in os.listdir(config_path):
                os.remove(os.path.join(config_path, file))
            os.rmdir(config_path)
        
        return True
    except Exception as e:
        print(f"Error al eliminar el perfil: {e}")
        return False

def load_profile(profile_name: str) -> Optional[Dict]:
    """
    Carga un perfil de configuración desde un archivo JSON basado en el nombre del perfil.

    Args:
        profile_name (str): Nombre del perfil para cargar la configuración.

    Returns:
        dict: Diccionario con la configuración del perfil si se encuentra el archivo, None si no se encuentra.
    """
    profile_path = f"profiles/{profile_name}.json"
    if not os.path.exists(profile_path):
        print(f"Perfil {profile_name} no encontrado.")
        return None
    
    with open(profile_path, 'r') as file:
        profile = json.load(file)
    return profile

