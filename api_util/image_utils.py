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

def convert_video():
    stream = ffmpeg.input('..\\..\\..\\tmp\\video\\input\\15_10_2024_10_28_38_LEFT.webm')
    stream = ffmpeg.hflip(stream)
    stream = ffmpeg.output(stream, '..\\..\\..\\tmp\\video\\output\\15_10_2024_10_28_38_LEFT.avi')
    ffmpeg.run(stream)
    # Python Check if the file exists
    if os.path.exists("..\\..\\..\\tmp\\video\\output\\15_10_2024_10_28_38_LEFT.avi"):
        print("File exists!")
        return True
    else:
        print("File does not exist.")
        return False
    