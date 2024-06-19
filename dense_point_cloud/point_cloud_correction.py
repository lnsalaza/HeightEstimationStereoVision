import csv
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import open3d as o3d
import numpy as np

def visualize_dense_point_cloud(pcd_file, model, show_marks = False, width=200, height=100, depth=1000, interval=100):
    if(show_marks):
        mark_lines = create_mark_lines(width, height, depth, interval)

    # Leer la nube de puntos densa
    pcd = o3d.io.read_point_cloud(pcd_file)

    # ESCALADO BETA
    scale_factor = 1.0
    scaling_matrix = np.eye(4)
    scaling_matrix[:3, :3] *= scale_factor

    pcd = pcd.transform(scaling_matrix)

    z_min, z_max = 0, 1000
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-float('inf'), -float('inf'), z_min), max_bound=(float('inf'), float('inf'), z_max))
    cropped_pcd = pcd.crop(bounding_box)

    # Obtener las coordenadas actuales de la nube de puntos
    points = np.asarray(cropped_pcd.points)

    Xy = points[:,:2]
    # Predecir las coordenadas Z corregidas usando el modelo
    z = points[:, 2].reshape(-1,1)  # Tomar solo las coordenadas X, y
    print(points)
    z_pred = model.predict(z)
    
    # Actualizar las coordenadas Z de la nube de puntos con las predicciones corregidas
    corrected_points = np.column_stack((Xy, z_pred))
    cropped_pcd.points = o3d.utility.Vector3dVector(corrected_points)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])

    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    # Agregar la geometría
    viewer.add_geometry(cropped_pcd)
    viewer.add_geometry(origin)
    if(show_marks):
        viewer.add_geometry(mark_lines)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 1

    # Ejecutar el visualizador
    viewer.run()

    # Destruir la ventana del visualizador
    viewer.destroy_window()

def create_mark_lines(width=200, height=100, depth=1000, interval=100):


    points = [
        [-width, -height, 0], [width,-height, 0], [-width, height, 0], [width, height, 0],
        [-width, -height, depth], [width, -height, depth], [-width, height, depth], [width, height, depth]
    ]


    for z in range(interval, depth, interval):
        points.append([-width, -height, z])
        points.append([width, -height, z])            
        points.append([-width, height, z])        
        points.append([width, height, z])      
        
    lines = [
        [0, 1],  [1, 3], [3, 2], [2, 0],
        [4, 5], [5, 7], [7, 6], [6, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    num_outer_points = 8
    for i in range(num_outer_points, len(points), 4):
        lines.append([i, i+1])
        lines.append([i+2, i+3])
    for i in range(num_outer_points + 4 * ((width // interval) - 1), len(points), 4):
        lines.append([i, i+1])
        lines.append([i+2, i+3])
    colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



if __name__ == "__main__":

    folder = "matlab_1_keypoint_disparity"
    situacion = "matlab_1_250_500"
    model_path = "../datasets/models/z_estimation_matlab_1_keypoint_ln_model.pkl"

    # Cargar el modelo de regresión lineal entrenado
    model = joblib.load(model_path)

    # Visualización de la nube de puntos densa
    dense_pcd_file = f"./point_clouds/{folder}/{situacion}_dense.ply"
    visualize_dense_point_cloud(dense_pcd_file, model, False)
