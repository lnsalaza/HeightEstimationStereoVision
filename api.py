from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from api_util.profile_management import *
from api_util.image_utils import *
from calibration.calibration import load_stereo_parameters, stereo_rectify, create_rectify_map, save_stereo_maps
from dense_point_cloud.point_cloud import *
import uvicorn
from fastapi import APIRouter
from typing import List, Dict




app = FastAPI(title="Stereo Calibration API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Los orígenes que deseas permitir (puedes usar ["*"] para todos)
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Cabeceras permitidas
)



@app.post("/upload_calibration/")
async def upload_calibration(file: UploadFile = File(...), profile_name: str = Form(...)):
    """
    Endpoint para subir un archivo JSON de calibración y generar el perfil de calibración correspondiente.

    Args:
        file (UploadFile): Archivo JSON que contiene los datos de calibración estéreo. Debe seguir un esquema específico.
        profile_name (str): Nombre del perfil de calibración bajo el cual se guardará la configuración.

    Returns:
        JSONResponse: Mensaje indicando éxito con las rutas del perfil y los mapas estéreo generados.

    Raises:
        HTTPException: Si el archivo no es un JSON o si hay un problema en el proceso de guardado o generación.
    """
    if file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="El archivo debe ser un JSON.")

    try:
        contents = await file.read()
        calibration_data = json.loads(contents)
        
        if 'cameraMatrix1' not in calibration_data:
            raise ValueError("Datos de calibración incompletos.")

        config_dir = f'config_files/{profile_name}'
        os.makedirs(config_dir, exist_ok=True)
        json_path = os.path.join(config_dir, 'calibration_data.json')
        with open(json_path, 'w') as json_file:
            json.dump(calibration_data, json_file)

        profile_data = generate_profile_data(calibration_data, profile_name)
        profile_path = save_profile(profile_data, profile_name)

        parameters = load_stereo_parameters(json_path)
        rectification = stereo_rectify(parameters)
        stereo_maps = {
            'Left': create_rectify_map(parameters['cameraMatrix1'], parameters['distCoeffs1'], rectification[0], rectification[2], parameters['imageSize']),
            'Right': create_rectify_map(parameters['cameraMatrix2'], parameters['distCoeffs2'], rectification[1], rectification[3], parameters['imageSize'])
        }
        xml_path = os.path.join(config_dir, 'stereo_map.xml')
        save_stereo_maps(xml_path, stereo_maps, rectification[4])

        return {
            "message": "Archivo de calibración subido y procesado con éxito",
            "profile_path": profile_path,
            "xml_path": xml_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_profiles/", response_model=list)
def get_profiles():
    """
    Devuelve una lista de todos los perfiles de calibración disponibles.

    Este endpoint escanea la carpeta 'profiles' para buscar archivos JSON que representen perfiles de calibración. 
    Cada perfil se devuelve con su nombre y la ruta de acceso al archivo correspondiente.

    Returns:
        list: Una lista de diccionarios, cada uno representando un perfil con su 'name' y 'path'.

    Raises:
        HTTPException: Si no se encuentran perfiles o si ocurre un error durante la operación de búsqueda.
    """
    try:
        profiles = list_profiles("profiles")  # Asegúrate de que la carpeta 'profiles' exista o maneja la excepción
        if not profiles:
            raise HTTPException(status_code=404, detail="No profiles found")
        return profiles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/delete_profile/{profile_name}")
def delete_profile_endpoint(profile_name: str):
    """
    Endpoint para eliminar un perfil y sus archivos asociados.

    Args:
        profile_name (str): El nombre del perfil a eliminar.
    
    Returns:
        dict: Mensaje sobre el resultado de la operación.
    """
    success = delete_profile(profile_name)
    if not success:
        raise HTTPException(status_code=404, detail="Profile not found or error deleting files")
    return {"message": f"Profile {profile_name} deleted successfully"}



@app.post("/generate_point_cloud/dense/")
async def generate_dense_point_cloud(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...)
):
    """
    Recibe dos imágenes estéreo, las rectifica utilizando el perfil de calibración especificado,
    y luego genera una nube de puntos 3D densa.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'RAFT', 'SELECTIVE').

    Returns:
        dict: Contiene la nube de puntos y los colores correspondientes, junto con el perfil y método usados.

    Raises:
        HTTPException: Si no se puede procesar la solicitud.
    """
    try:
        left_image = await read_image_from_upload(img_left)
        right_image = await read_image_from_upload(img_right)

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)

        # Generar nube de puntos
        point_cloud, colors = generate_dense_point_cloud(left_image_rect, right_image_rect, profile, method, use_max_disparity=True)
        return {
            "point_cloud": point_cloud.tolist(),
            "colors": colors.tolist(),
            "profile_used": profile_name,
            "method_used": method
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

