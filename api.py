from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import DBSCAN
from typing import Optional, List, Dict
from dense_point_cloud import PointCloudConfig
from dense_point_cloud import point_cloud as pc
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Point Cloud API"}

@app.get("/point_cloud")
async def get_dense_point_cloud(
    image_l: UploadFile = File(..., description="Imagen izquierda (formato: PNG, JPG)"),
    image_r: UploadFile = File(..., description="Imagen derecha (formato: PNG, JPG)"),
    method: str = "SGBM",
    apply_filter: bool = False,
    custom_mask_filter: Optional[str] = None,
    calibration_params: Dict[str, float] = None
):
    """
    Genera una nube de puntos densa a partir de dos imágenes utilizando el método especificado.
    
    Args:
        image_l (UploadFile): Imagen izquierda.
        image_r (UploadFile): Imagen derecha.
        method (str): Método de disparidad a utilizar ("SGBM", "RAFT", "SELECTIVE").
        apply_filter (bool): Si se debe aplicar un filtro a la nube de puntos.
        custom_mask_filter (Optional[str]): Máscara de filtro personalizada, si es necesario.
        calibration_params (Dict[str, float]): Parámetros de calibración.
        
    Returns:
        JSONResponse: Contiene la nube de puntos y metadatos adicionales.
    """
    try:
        if calibration_params is None:
            raise HTTPException(status_code=400, detail="calibration_params is required")

        required_keys = ["fx", "fy", "cx1", "cx2", "cy", "baseline", "Q"]
        for key in required_keys:
            if key not in calibration_params:
                raise HTTPException(status_code=400, detail=f"{key} is required in calibration_params")

        # Leer las imágenes de entrada
        image_l_bytes = await image_l.read()
        image_r_bytes = await image_r.read()

        # Guardar las imágenes en archivos temporales
        image_l_path = "/tmp/image_l.png"
        image_r_path = "/tmp/image_r.png"
        with open(image_l_path, 'wb') as f:
            f.write(image_l_bytes)
        with open(image_r_path, 'wb') as f:
            f.write(image_r_bytes)

        # Crear la configuración para la nube de puntos
        config = PointCloudConfig(
            calibration_params=calibration_params,
            method=method,
            apply_correction=apply_filter,
            is_roi=(custom_mask_filter == "roi"),
            model_path='../datasets/models/matlab_1/height_lr.pkl'
        )

        # Generar la nube de puntos
        dense_point_cloud, dense_colors, metadata = pc.generate_point_cloud(
            image_l_path, image_r_path, method, apply_filter, config
        )

        # Convertir la nube de puntos a una lista de puntos para el JSON de respuesta
        points = np.asarray(dense_point_cloud).tolist()
        colors = np.asarray(dense_colors).tolist()

        response = {
            "name": "point_cloud_type",
            "points": points,
            "colors": colors,
            "metadata": metadata
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))