import os
import cv2
import numpy as np
import ffmpeg
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

def create_video_from_frames(image_folder: str, criteria: str, output_video_path: str, fps: int = 30):
    image_files = glob(os.path.join(image_folder, f"*{criteria}.png"))
    if not image_files:
        raise ValueError("No se encontraron imagenes para procesar")
    sorted_images=natsorted(image_files)
    frame = cv2.imread(sorted_images[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in sorted_images:
        frame = cv2.imread(image_file)
        video_writer.write(frame)
    
    video_writer.release()

def convert_video(input_path: str, output_path: str, codec: str = 'XVID') -> None:
    """
    Convierte un video de formato webm a avi utilizando OpenCV.

    Args:
        input_path (str): Ruta del archivo de video de entrada.
        output_path (str): Ruta donde se guardará el archivo de video de salida.
        codec (str): Codec a utilizar para la codificación del video de salida.
    """
    # Abrir el video de entrada con OpenCV
    cap = cv2.VideoCapture(input_path)

    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        raise Exception("Error al abrir el video de entrada.")

    # Obtener las propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Establecer FPS por defecto si no se puede obtener

    # Definir el codec y crear el objeto VideoWriter para el archivo de salida
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Procesar el video frame por frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Escribir el frame en el archivo de salida
        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    