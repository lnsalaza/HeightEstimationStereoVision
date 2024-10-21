import os
import cv2
import shutil
import numpy as np
from glob import glob
from natsort import natsorted
from fastapi import UploadFile

async def read_image_from_upload(file: UploadFile) -> np.array:
    """
    Lee una imagen desde un archivo subido y la convierte en un array de numpy.

    Args:
        file (UploadFile): Imagen subida.

    Returns:
        np.array: Imagen como array de numpy.
    """
    image_contents = await file.read()
    image_array = np.frombuffer(image_contents, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def convert_video(input_path: str, output_path: str, codec: str = 'XVID', fps: float = 30.0) -> None:
    """
    Convierte un video de formato webm a avi utilizando OpenCV.

    Args:
        input_path (str): Ruta del archivo de video de entrada.
        output_path (str): Ruta donde se guardará el archivo de video de salida.
        codec (str): Codec a utilizar para la codificación del video de salida.
        fps (float): Tasa de cuadros por segundo para el video de salida.
    """
    # Abrir el video de entrada con OpenCV
    cap = cv2.VideoCapture(input_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        raise Exception("Error al abrir el video de entrada.")

    # Obtener las propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Opcional: Validar los FPS obtenidos
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0 or input_fps > 120:
        print(f"FPS inválidos detectados ({input_fps}). Se establecerán a {fps} FPS.")
    else:
        fps = input_fps

    # Definir el codec y crear el objeto VideoWriter para el archivo de salida
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Procesar el video frame por frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Escribir el frame en el archivo de salida
        out.write(frame)
        frame_count += 1

    print(f"Conversión completada: {frame_count} frames escritos a {fps} FPS.")

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def delete_tmp_folder(path: str):
    """
    Elimina un directorio temporal si existe.

    Args:
        path (str): Ruta del directorio a eliminar.
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    