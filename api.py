import io
import os
import json
import uuid

from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
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
from dense_point_cloud.point_cloud_alt import *
from dense_point_cloud.util import (
    convert_point_cloud_format, 
    convert_individual_point_clouds_format
)

app = FastAPI(title="Stereo Vision API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Los orígenes que se desea permitir (Usar ["*"] para todos)
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Cabeceras permitidas
)


@app.get("/")
async def root():
    return {"message": "Te has conectado exitosamente. Bienvenido al API de Stereo Vision, todo parece estar funcionando bien!"}

@app.post("/add_profile/")
async def add_profile(file: UploadFile = File(...), profile_name: str = Form(...)):
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

        

        parameters = load_stereo_parameters(json_path)
        rectification = stereo_rectify(parameters)
        stereo_maps = {
            'Left': create_rectify_map(parameters['cameraMatrix1'], parameters['distCoeffs1'], rectification[0], rectification[2], parameters['imageSize']),
            'Right': create_rectify_map(parameters['cameraMatrix2'], parameters['distCoeffs2'], rectification[1], rectification[3], parameters['imageSize'])
        }
        xml_path = os.path.join(config_dir, 'stereo_map.xml')
        save_stereo_maps(xml_path, stereo_maps, rectification[4])

        profile_data = generate_profile_data(calibration_data, profile_name, rectification[4])
        profile_path = save_profile(profile_data, profile_name)

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

@app.get("/get_profile/{profile_name}", response_model=Optional[Dict])
def get_profile(profile_name: str):
    """
    Obtiene el JSON de un perfil específico basado en el nombre del perfil.

    Args:
        profile_name (str): Nombre del perfil a obtener.

    Returns:
        dict: JSON del perfil si se encuentra, de lo contrario se devuelve un error 404.

    Raises:
        HTTPException: Si el perfil no se encuentra, se devuelve un error 404.
    """
    profile = load_profile(profile_name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")
    return profile

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
        # Leer las imágenes
        if method != "realsense":
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right)
        else:
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right, image_type="depth")

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        if method != "realsense":
            left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)
        else:
            left_image_rect, right_image_rect = left_image, right_image
            
        # Generar nube de puntos
        point_cloud, colors = generate_dense_point_cloud(left_image_rect, right_image_rect, profile, method, use_max_disparity=use_max_disparity, normalize=normalize)
        return {
            "point_cloud": [point_cloud.tolist()],
            "colors": [colors.tolist()],
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
        # Leer las imágenes
        if method != "realsense":
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right)
        else:
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right, image_type="depth")

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        if method != "realsense":
            left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)
        else:
            left_image_rect, right_image_rect = left_image, right_image

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
        # Leer las imágenes
        if method != "realsense":
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right)
        else:
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right, image_type="depth")

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        if method != "realsense":
            left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)
        else:
            left_image_rect, right_image_rect = left_image, right_image

        # Generar listas de nubes de puntos para cada objeto detectado
        point_clouds_list, colors_list, keypoints3d, max_coords = generate_individual_filtered_point_clouds(
            left_image_rect, right_image_rect, profile, method, use_roi, use_max_disparity, normalize
        )
        
        return {
            "point_clouds": [pc.tolist() for pc in point_clouds_list],
            "colors": [colors.tolist() for colors in colors_list],
            "keypoints_3d": [kp.tolist() for kp in keypoints3d],
            "profile_used": profile_name,
            "method_used": method,
            "roi_used": use_roi,
            "normalized": normalize,
            "max_coords": max_coords
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
        if method != "realsense":
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right)
        else:
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right, image_type="depth")

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        if method != "realsense":
            left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)
        else:
            left_image_rect, right_image_rect = left_image, right_image

        # Generar nubes de puntos individuales
        point_clouds_list, colors_list, keypoints3d_list, _ = generate_individual_filtered_point_clouds(
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

@app.post("/face/height_estimation/")
async def estimate_height_from_face(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...)
):
    """
    Estima la altura de una persona en la escena usando las medidas/proporciones del rostro.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
    Returns:
        JSONResponse: Contiene la altura estimada y la profundidad a la que se encuentra la persona.
    """
    try:
        
        # Leer las imágenes subidas
        left_image = await read_image_from_upload(img_left)
        right_image = await read_image_from_upload(img_right)

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)


        # Se estima la altura y la profundidad de la persona
        height, depth = estimate_height_from_face_proportions(img_left=left_image_rect, img_right=right_image_rect, config=profile)
        

        # Se asume que hay al menos una nube de puntos de la cual estimar la altura
        if height < 0:
            raise HTTPException(status_code=404, detail="No se pudo estimar la altura de la persona de forma correcta.")

        return {    
            "profile_used": profile_name,
            "height": height,
            "depth": depth
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/face/eyes_camera_separation/")
async def estimate_difference_eyes_camera(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...)
):
    """
    Estima la separación en Y entre los ojos de la persona en la escena y el centro optico de la camara.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
    Returns:
        JSONResponse: Contiene la separación estimada y la profundidad a la que se encuentra la persona.
    """
    try:
        
        # Leer las imágenes subidas
        left_image = await read_image_from_upload(img_left)
        right_image = await read_image_from_upload(img_right)

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)


        # Se estima la altura y la profundidad de la persona
        height, depth = estimate_separation_eyes_camera(img_left=left_image_rect, img_right=right_image_rect, config=profile)
        

        # Se asume que hay al menos una nube de puntos de la cual estimar la altura
        if height < 0:
            raise HTTPException(status_code=404, detail="No se pudo estimar la separacíon entre los ojos de la persona y el centro optico de la camara de forma correcta.")

        return {    
            "profile_used": profile_name,
            "height": height,
            "depth": depth
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.post("/generate_point_cloud/nodense/features/")
async def generate_point_cloud_with_features(
    img_left: UploadFile = File(...),
    img_right: UploadFile = File(...),
    profile_name: str = Form(...),
    method: str = Form(...),
    use_max_disparity: bool = True,
    normalize: bool =True
):
    """
    Genera nubes de puntos, keypoints 3D y extrae características para cada persona detectada a partir de imágenes estéreo.

    Args:
        img_left (UploadFile): Imagen del lado izquierdo como archivo subido.
        img_right (UploadFile): Imagen del lado derecho como archivo subido.
        profile_name (str): Nombre del perfil de calibración a utilizar.
        method (str): Método de disparidad a utilizar ('SGBM', 'WLS-SGBM', 'RAFT', 'SELECTIVE').
        use_max_disparity (bool): Indica si activar o desactivar el uso de la disparidad máxima para optimizar la nube de puntos.
        normalize (bool): Indica si se debe normalizar la nube de puntos.

    Returns:
        dict: Contiene las nubes de puntos, colores, keypoints 3D y características extraídas para cada persona detectada.
    
    Raises:
        HTTPException: Si hay un error en el procesamiento.
    """
    try:
        # Leer imágenes cargadas
        if method != "realsense":
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right)
        else:
            left_image = await read_image_from_upload(img_left)
            right_image = await read_image_from_upload(img_right, image_type="depth")

        # Cargar la configuración del perfil y rectificar las imágenes
        profile = load_profile(profile_name)
        if not profile:
            raise HTTPException(status_code=404, detail=f"Perfil {profile_name} no encontrado.")

        if method != "realsense":
            left_image_rect, right_image_rect = rectify_images(left_image, right_image, profile_name)
        else:
            left_image_rect, right_image_rect = left_image, right_image
        
        # Generar nubes de puntos, colores, keypoints y extraer características
        point_clouds, colors, keypoints, features, max_coords = generate_filtered_point_cloud_with_features(
            left_image_rect, right_image_rect, profile, method, use_roi=False, use_max_disparity=use_max_disparity, normalize=normalize
        )
        
        return {    
            "profile_used": profile_name,
            "method_used": method,
            "features": features,
            "max_coords": max_coords
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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

@app.post("/convert_video/")
async def convert_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    fps: float = Form(...),
):
    """
    Convierte un archivo de video subido a formato AVI con un codec específico y el número de fotogramas por segundo (fps) indicado.

    Args:
        background_tasks (BackgroundTasks): Manejador de tareas en segundo plano.
        file (UploadFile): Archivo de video subido por el usuario, soporta formatos "video/webm", "video/avi" y "video/webm;codecs=vp9".
        fps (float): Número de fotogramas por segundo deseado para el video de salida.

    Returns:
        FileResponse: Respuesta con el archivo de video convertido en formato AVI. 
                      Si ocurre un error, se lanza una excepción HTTP con el código de error y detalle.
    """
    try:
        # Validar el tipo de archivo
        if file.content_type not in ["video/webm", "video/avi", 'video/webm;codecs=vp9']:
            raise HTTPException(status_code=400, detail="Formato de archivo no soportado.")

        # Directorio temporal personalizado
        tmp_dir = f'../tmp/video/{uuid.uuid4()}'
        os.makedirs(tmp_dir, exist_ok=True)

        # Generar nombres de archivos únicos
        input_filename = "input.webm"
        output_filename = "output.avi"

        input_path = os.path.join(tmp_dir, input_filename)
        output_path = os.path.join(tmp_dir, output_filename)

        # Guardar el archivo subido en el directorio tmp
        with open(input_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Llamar a la función de conversión de video con el codec deseado
        convert_video(input_path, output_path, codec='XVID', fps=fps)

        # Verificar que el archivo de salida se haya creado
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Error al crear el archivo de salida.")

        # Agregar tarea en segundo plano para eliminar el directorio temporal después de la respuesta
        background_tasks.add_task(delete_tmp_folder, tmp_dir)

        # Devolver el archivo convertido al cliente
        return FileResponse(
            output_path,
            media_type='video/avi',
            filename=f"{os.path.splitext(file.filename)[0]}.avi"
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))