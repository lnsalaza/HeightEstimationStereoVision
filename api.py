import os
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from api_util.profile_management import *
from api_util.image_utils import *
from calibration.calibration import (
    load_stereo_parameters, 
    stereo_rectify, 
    create_rectify_map, 
    save_stereo_maps
)
from dense_point_cloud.point_cloud import *
from dense_point_cloud.util import (
    convert_point_cloud_format, 
    convert_individual_point_clouds_format
)

app = FastAPI(title="Stereo Calibration API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Los orígenes que se desea permitir (Usar ["*"] para todos)
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
        profiles = list_profiles("profiles") 
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
async def dense_point_cloud(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...),
    use_max_disparity: bool = False,
    normalize: bool = True  
):
    """
    Recibe dos imágenes estéreo, las rectifica utilizando el perfil de calibración especificado,
    y luego genera una nube de puntos 3D densa utilizando el método de disparidad seleccionado.
    Opcionalmente, la nube de puntos puede ser normalizada a una escala de unidad estándar.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
        use_max_disparity (bool): Indica si se activa o desactiva un filtrado de puntos flotantes.
        normalize (bool): Indica si se debe normalizar la nube de puntos a una escala de unidad estándar.

    Returns:
        dict: Contiene la nube de puntos y los colores correspondientes, junto con el perfil y método usados, además del estado de la normalización.

    Raises:
        HTTPException: Si no se puede procesar la solicitud debido a errores en la carga de perfiles, en la rectificación de imágenes o en la generación de la nube de puntos.
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
        point_cloud, colors = generate_dense_point_cloud(left_image_rect, right_image_rect, profile, method, use_max_disparity=use_max_disparity, normalize=normalize)
        return {
            "point_cloud": point_cloud.tolist(),
            "colors": colors.tolist(),
            "profile_used": profile_name,
            "method_used": method,
            "normalized": normalize
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_point_cloud/nodense/complete/")
async def complete_no_dense_point_cloud(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...),
    use_roi: bool = True,
    use_max_disparity: bool = True,
    normalize: bool = True  
):
    """
    Recibe dos imágenes estéreo, las rectifica utilizando el perfil de calibración especificado,
    y luego genera una nube de puntos 3D filtrada y combinada utilizando el método de disparidad seleccionado.
    Opcionalmente, la nube de puntos puede ser normalizada a una escala de unidad estándar.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
        use_roi (bool): Indica si aplicar una región de interés (ROI) durante el procesamiento.
        use_max_disparity (bool): Indica si se activa o desactiva un filtrado de puntos flotantes.
        normalize (bool): Indica si se debe normalizar la nube de puntos a una escala de unidad estándar.

    Returns:
        dict: Contiene la nube de puntos filtrada y los colores correspondientes, junto con el perfil y método usados, además del estado de la normalización.

    Raises:
        HTTPException: Si no se puede procesar la solicitud debido a errores en la carga de perfiles, en la rectificación de imágenes o en la generación de la nube de puntos.
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
        point_cloud, colors = generate_combined_filtered_point_cloud(left_image_rect, right_image_rect, profile, method, use_roi, use_max_disparity, normalize=normalize)
        return {
            "point_cloud": point_cloud.tolist(),
            "colors": colors.tolist(),
            "profile_used": profile_name,
            "method_used": method,
            "roi_used": use_roi,
            "normalized": normalize
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.post("/generate_point_cloud/nodense/individual/")
async def individual_no_dense_point_cloud(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...),
    use_roi: bool = True,
    use_max_disparity: bool = True,
    normalize: bool = True    # Nuevo parámetro para controlar la normalización
):
    """
    Recibe dos imágenes estéreo, las rectifica utilizando el perfil de calibración especificado,
    y luego genera listas separadas de nubes de puntos 3D y colores para cada objeto detectado individualmente,
    utilizando el método de disparidad seleccionado. Opcionalmente, las nubes de puntos pueden ser normalizadas.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
        use_roi (bool): Indica si aplicar una Región de Interés (ROI) durante el procesamiento.
        use_max_disparity (bool): Indica si utilizar la disparidad máxima para optimizar la nube de puntos.
        normalize (bool): Indica si se debe normalizar la nube de puntos a una escala de unidad estándar.

    Returns:
        dict: Contiene listas de nubes de puntos y colores, cada una correspondiente a un objeto detectado individualmente, junto con el perfil y método usados, además del estado de la normalización.

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

        # Generar listas de nubes de puntos para cada objeto detectado
        point_clouds_list, colors_list, keypoints3d = generate_individual_filtered_point_clouds(
            left_image_rect, right_image_rect, profile, method, use_roi, use_max_disparity, normalize
        )
        
        return {
            "point_clouds": [pc.tolist() for pc in point_clouds_list],
            "colors": [colors.tolist() for colors in colors_list],
            "keypoints_3d": [kp.tolist() for kp in keypoints3d],
            "profile_used": profile_name,
            "method_used": method,
            "roi_used": use_roi,
            "normalized": normalize
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_point_cloud/nodense/height_estimation/")
async def estimate_height_from_cloud(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...),
    use_max_disparity: bool = True,
    normalize: bool = True,  # Parámetro para controlar la normalización
    k: int = Form(5),
    threshold_factor: float = Form(1.0),
    m_initial: float = Form(50.0)
):
    """
    Genera nubes de puntos individuales para cada persona detectada en las imágenes estéreo
    y estima la altura de cada persona desde la nube de puntos correspondiente.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
        use_max_disparity (bool): Indica si se activa o desactiva un filtrado de puntos flotantes.
        normalize (bool): Indica si se normaliza la nube de puntos a una escala de unidad estándar.
        k (int): Número de vecinos más cercanos para calcular el centroide.
        threshold_factor (float): Factor de umbral para eliminar el ruido en el cálculo del centroide.
        m_initial (float): Rango inicial para filtrar los puntos alrededor del centroide.

    Returns:
        JSONResponse: Contiene la altura estimada y el centroide calculado para cada persona.
    """
    try:
        # Forzar use_roi a False para que el proceso se limite a personas individuales
        use_roi = False

        # Leer las imágenes subidas
        left_image = await read_image_from_upload(img_left)
        right_image = await read_image_from_upload(img_right)

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)

        # Generar nubes de puntos individuales
        point_clouds_list, colors_list, keypoints3d_list = generate_individual_filtered_point_clouds(
            left_image_rect, right_image_rect, profile, method, use_roi, use_max_disparity, normalize
        )

        # Se asume que hay al menos una nube de puntos de la cual estimar la altura
        if not point_clouds_list:
            raise HTTPException(status_code=404, detail="No se encontraron nubes de puntos individuales.")

        # Procesar la altura de todas las personas detectadas
        results = {}
        for idx, point_cloud in enumerate(point_clouds_list):
            height, centroid = estimate_height_from_point_cloud(
                point_cloud, k=k, threshold_factor=threshold_factor, m_initial=m_initial
            )
            if height is not None:
                results[f"person_{idx + 1}"] = {
                    "height": height,
                    "centroid": centroid.tolist()
                }
            else:
                results[f"person_{idx + 1}"] = {
                    "message": "No se pudo estimar la altura para esta persona.",
                    "centroid": centroid.tolist()
                }

        # Agregar información sobre el perfil y el método utilizado
        results["profile_used"] = profile_name
        results["method_used"] = method

        return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la nube de puntos: {str(e)}")

@app.post("/download_point_cloud/dense/")
async def download_converted_point_cloud(format: str):
    """
    Convierte el archivo temporal de la nube de puntos densa al formato solicitado y lo envía como descarga.

    Args:
        format (str): Formato deseado para la conversión ('ply', 'xyz', 'pcd', 'pts', 'xyzrgb').

    Returns:
        FileResponse: Archivo convertido para descargar o mensaje de error si algo falla.
    """
    try:
        # Realizar la conversión del archivo temporal al formato solicitado
        success, converted_file_path = convert_point_cloud_format(output_format=format)
        
        if not success:
            raise HTTPException(status_code=500, detail=converted_file_path)

        # Verificar que el archivo convertido existe
        if not os.path.exists(converted_file_path):
            raise HTTPException(status_code=404, detail="El archivo convertido no se encontró.")

        # Devolver el archivo convertido como una descarga
        return FileResponse(
            path=converted_file_path,
            filename=os.path.basename(converted_file_path),
            media_type='application/octet-stream'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la descarga: {str(e)}")
    


@app.post("/download_point_cloud/nodense/")
async def download_individual_converted_point_clouds(format: str):
    """
    Convierte las nubes de puntos individuales a un formato especificado, las comprime junto con los keypoints en un archivo ZIP,
    y lo envía como descarga.

    Args:
        format (str): Formato deseado para la conversión ('ply', 'xyz', 'pcd', 'pts', 'xyzrgb').

    Returns:
        FileResponse: Archivo ZIP convertido para descargar o mensaje de error si algo falla.
    """
    try:
        # Realizar la conversión y compresión de los archivos temporales al formato solicitado
        success, zip_file_path = convert_individual_point_clouds_format(output_format=format)
        
        if not success:
            raise HTTPException(status_code=500, detail=zip_file_path)

        # Verificar que el archivo ZIP convertido existe
        if not os.path.exists(zip_file_path):
            raise HTTPException(status_code=404, detail="El archivo ZIP convertido no se encontró.")

        # Devolver el archivo ZIP convertido como una descarga
        return FileResponse(
            path=zip_file_path,
            filename=os.path.basename(zip_file_path),
            media_type='application/zip'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la descarga: {str(e)}")
