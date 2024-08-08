import cv2
import numpy as np
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
