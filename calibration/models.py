from pydantic import BaseModel, Field
from typing import List

class StereoCalibrationData(BaseModel):
    cameraMatrix1: List[List[float]]
    distCoeffs1: List[float]
    cameraMatrix2: List[List[float]]
    distCoeffs2: List[float]
    imageSize: List[int]
    stereoR: List[List[float]]
    stereoT: List[float]
    flCamera1: List[float]
    flCamera2: List[float]

    class Config:
        schema_extra = {
            "example": {
                "cameraMatrix1": [
                    [1031.1507131471858, 0, 0],
                    [0, 1032.117568616037, 0],
                    [934.1223953047944, 534.1022934087931, 1]
                ],
                "distCoeffs1": [-0.016201627022114046, -0.010799821417060772, 0, 0],
                "cameraMatrix2": [
                    [1036.538061496097, 0, 0],
                    [0, 1037.8570658155766, 0],
                    [942.4783723248192, 552.5119162086186, 1]
                ],
                "distCoeffs2": [-0.007828421163718173, -0.03970152227984051, 0, 0],
                "imageSize": [1080, 1920],
                "stereoR": [
                    [0.9999942687219076, -0.00008124413025362022, 0.003384659913294103],
                    [0.00009674211738303457, 0.999989511743685, -0.004578978443842968],
                    [-0.0033842523989922736, 0.00457927963961076, 0.9999837883854329]
                ],
                "stereoT": [-60.0630049839543, -0.24820751825046766, 1.0670042205575287],
                "flCamera1": [1031.1507131471858, 1032.117568616037],
                "flCamera2": [1036.538061496097, 1037.8570658155766]
            }
        }
