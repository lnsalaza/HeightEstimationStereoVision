import json
import cv2

import numpy as np
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from dense_point_cloud.point_cloud import * 


if __name__ == "__main__":
    # Cargar las imágenes como arrays
    img_left = cv2.imread("images/calibration_results/matlab_1/flexometer/250 y 600/14_13_13_13_05_2024_IMG_LEFT.jpg")
    img_right = cv2.imread("images/calibration_results/matlab_1/flexometer/250 y 600/14_13_13_13_05_2024_IMG_RIGHT.jpg")
    
    if img_left is None or img_right is None:
        raise FileNotFoundError("Una o ambas imágenes no pudieron ser cargadas. Verifique las rutas.")

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)


    # Cargar configuración desde el archivo JSON
    config = load_config("profiles/profile1.json")
    
    # Asumiendo que queremos usar el método SGBM, ajusta si es RAFT o SELECTIVE según tu configuración
    method = 'SELECTIVE'

    # #TEST MAPA DISPARIDAD
    # test_disparity_map(img_left, img_right, config, method)

    # #TEST NUBE DE PUNTOS DENSA
    #test_point_cloud(img_left, img_right, config, method, use_max_disparity=False, normalized=False)


    # #TEST NUBE DE PUNTOS NO DENSA TOTAL
    #test_filtered_point_cloud(img_left, img_right, config, method, use_roi=True, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA TOTAL
    # test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)



    # #TEST NUBE DE PUNTOS NO DENSA INDIVIDUAL
    test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi=False, use_max_disparity=True, normalized=False)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA INDIVIDUAL
    #test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    # points, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, False, True)
    # # Seleccionar un subconjunto aleatorio de puntos, incluyendo el origen
    # NUM_POINTS_TO_DRAW = len(points)
    # subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW - 1,), replace=False)
    # subset = np.append(subset, points.shape[0] - 1)  # Asegurar que el origen esté incluido
    # points_subset = points[subset]
    # colors_subset = colors[subset]

    # x, y, z = points_subset.T

    # fig = go.Figure(
    #     data=[
    #         go.Scatter3d(
    #             x=x, y=y, z=z, # flipped to make visualization nicer
    #             mode='markers',
    #             marker=dict(size=1, color=colors_subset)
    #         )
    #     ],
    #     layout=dict(
    #         scene=dict(
    #             xaxis=dict(visible=True),
    #             yaxis=dict(visible=True),
    #             zaxis=dict(visible=True),
    #         )
    #     )
    # )
    # fig.show()
    print("TESTING IS ENDING...")


