import json
import cv2

import numpy as np
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from dense_point_cloud.point_cloud import * 
from dense_point_cloud.util import convert_point_cloud_format, convert_individual_point_clouds_format
from testing_util import convert_to_gray
######################### NEXT FUNCTION ARE JUST FOR TESTING PURPOSES #################################
def test_disparity_map(img_left, img_right, config, method):
    # Calcular el mapa de disparidad
    disparity_map = compute_disparity(img_left, img_right, config, method)


    # Visualizar el mapa de disparidad generado
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

def test_point_cloud(img_left, img_right, config, method, use_max_disparity, normalized):
    # Generar la nube de puntos 3D
    point_cloud, colors = generate_dense_point_cloud(img_left, img_right, config, method, use_max_disparity, normalized)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/densaDEMO.ply")
    
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="3D Point Cloud", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])
    viewer.add_geometry(origin)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 1  # Establecer el tamaño de los puntos

    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity, normalized)
    convert_point_cloud_format(output_format='xyzrgb')
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Crea bounding boxes de los puntos en la nube
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    viewer.add_geometry(aabb)

    
    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    if not use_roi:
        opt.point_size = 5  # Establecer el tamaño de los puntos
    else:
        opt.point_size = 1
    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity, normalized)

    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Calcular los centroides de los clusters/personas
    centroids = compute_centroids(point_cloud)

    print(f"CENTROIDES: {centroids}")
    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud with Centroids", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Añadir las esferas de los centroides a la ventana de visualización
    for centroid in centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)  # Ajusta el radio según sea necesario
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo
        viewer.add_geometry(sphere)

    # Crear bounding boxes de los puntos en la nube
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    viewer.add_geometry(aabb)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    if not use_roi:
        opt.point_size = 5  # Establecer el tamaño de los puntos
    else:
        opt.point_size = 1
    
    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list, keypoints3d = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity, normalized)
    convert_individual_point_clouds_format(output_format='xyzrgb')
    for i, (point_cloud, colors) in enumerate(zip(point_cloud_list, color_list)):
        # Convertir los datos de la nube de puntos y colores a formato Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear una ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos a la ventana de visualización
        viewer.add_geometry(pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Establecer el tamaño de los puntos
        else:
            opt.point_size = 1
        # Ejecutar la visualización
        
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()
    
    for i, (point_cloud, colors) in enumerate(zip(keypoints3d, color_list)):
        # Convertir los datos de la nube de puntos y colores a formato Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear una ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos a la ventana de visualización
        viewer.add_geometry(pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Establecer el tamaño de los puntos
        else:
            opt.point_size = 1
        # Ejecutar la visualización
        
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()

def test_filtered_point_cloud_with_features(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    """
    Función de testing para generar nubes de puntos filtradas individuales junto con keypoints 3D y características, y luego visualizarlas.

    Args:
        img_left (np.array): Imagen del lado izquierdo como array de numpy.
        img_right (np.array): Imagen del lado derecho como array de numpy.
        config (dict): Diccionario de configuración del perfil.
        method (str): Método de disparidad (SGBM, RAFT, etc.).
        use_roi (bool): Indica si se usa una región de interés.
        use_max_disparity (bool): Indica si se utiliza la disparidad máxima.
        normalized (bool): Indica si la nube de puntos se normaliza.
    """
    # Generar nubes de puntos, colores, keypoints 3D y características
    point_cloud_list, color_list, keypoints3d, features = generate_filtered_point_cloud_with_features(
        img_left, img_right, config, method, use_roi, use_max_disparity, normalized
    )

    # Convertir las nubes de puntos individuales a formato XYZRGB
    convert_individual_point_clouds_format(output_format='xyzrgb')

    # Visualizar cada nube de puntos generada para los objetos detectados
    for i, (point_cloud, colors) in enumerate(zip(point_cloud_list, color_list)):
        # Crear objeto PointCloud de Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear la ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos a la ventana de visualización
        viewer.add_geometry(pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Tamaño de puntos sin ROI
        else:
            opt.point_size = 1  # Tamaño de puntos con ROI

        # Ejecutar la visualización
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()

    # Visualizar los keypoints 3D generados
    for i, (keypoints, colors) in enumerate(zip(keypoints3d, color_list)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Keypoints for Object {i+1}", width=800, height=600)
        viewer.add_geometry(pcd)

        opt = viewer.get_render_option()
        opt.point_size = 5 if not use_roi else 1

        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()

    # Mostrar características extraídas
    for idx, feature in enumerate(features):
        print(f"Features for Person {idx+1}: {feature}")

def test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list, keypoints3d = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity, normalized)
    
    for i, (point_cloud, colors) in enumerate(zip(point_cloud_list, color_list)):
        # Calcular el centroide omitiendo puntos ruidosos
        centroid = compute_centroid(point_cloud)
        print(f"CENTROIDE:  {centroid}")

        # Convertir los datos de la nube de puntos y colores a formato Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        # Crear una nube de puntos para el centroide (VARIANTE 1)
        # centroid_pcd = o3d.geometry.PointCloud()
        # centroid_pcd.points = o3d.utility.Vector3dVector(np.array([centroid]))
        # centroid_pcd.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0]]))  # Rojo

        # Crear una esfera para el centroide (VARIANTE 2)
        centroid_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=5)  # Ajusta el radio según sea necesario
        centroid_pcd.translate(centroid)
        centroid_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo

        # Crear una ventana de visualización
        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        # Añadir la nube de puntos y el centroide a la ventana de visualización
        viewer.add_geometry(pcd)
        viewer.add_geometry(centroid_pcd)

        # Configurar opciones de renderizado
        opt = viewer.get_render_option()
        if not use_roi:
            opt.point_size = 5  # Establecer el tamaño de los puntos
        else:
            opt.point_size = 1
        
        # Ejecutar la visualización
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()

def test_estimate_height_from_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity, normalized):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list, keypoints3d = generate_individual_filtered_point_clouds(
        img_left, img_right, config, method, use_roi, use_max_disparity, normalized
    )
    i = 0
    for (point_cloud, colors) in zip(point_cloud_list, color_list):
        # Estimar la altura de la persona
        estimated_height, centroid = estimate_height_from_point_cloud(point_cloud=point_cloud, m_initial=100)
        estimated_height
        if estimated_height is not None:
            print(f"Con el centroide {centroid}.\n'Altura estimada de la persona {i+1}: {estimated_height:.2f} unidades")
        else:
            print(f"No se pudo estimar la altura de la persona {i+1}")

        # Visualización de la nube de puntos con el centroide y keypoints
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

        centroid_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=5)  # Ajusta el radio según sea necesario
        centroid_pcd.translate(centroid)
        centroid_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Rojo

        viewer = o3d.visualization.Visualizer()
        viewer.create_window(window_name=f"3D Point Cloud for Object {i+1}", width=800, height=600)

        viewer.add_geometry(pcd)
        viewer.add_geometry(centroid_pcd)

        opt = viewer.get_render_option()
        opt.point_size = 1 if use_roi else 5
        i = i + 1
        viewer.run()
        viewer.clear_geometries()
        viewer.destroy_window()

def test_estimate_height_from_face_proportions(img_left, img_right, config):
    depth = estimate_height_from_face_proportions(img_left=img_left, img_right=img_right, config=config)
    print(depth)


def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":

    import torch

    print(torch.cuda.is_available())


    # # Cargar las imágenes como arrays
    # img_left = cv2.imread("images/laser/groundTruth/298 y 604/15_22_21_07_06_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("images/laser/groundTruth/298 y 604/15_22_21_07_06_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("../originals/laser/groundTruth/790/15_30_41_07_06_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("../originals/laser/groundTruth/790/15_30_41_07_06_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("images/distances/300/14_06_19_13_05_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("images/distances/300/14_06_19_13_05_2024_IMG_RIGHT.jpg")

    
    # img_left = cv2.imread("images/distances/300/14_06_13_13_05_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("images/distances/300/14_06_13_13_05_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("images/distances/400/14_07_35_13_05_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("images/distances/400/14_07_35_13_05_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("images/laser/calibracion/300/14_47_48_07_06_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("images/laser/calibracion/300/14_47_48_07_06_2024_IMG_RIGHT.jpg")
                                                                                
    # img_left = cv2.imread("../HeightEstimationStereoVision/images/distances/250 y 600/14_13_13_13_05_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("../HeightEstimationStereoVision/images/distances/250 y 600/14_13_13_13_05_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("../originals/h_train/457_z/173/14_32_27_31_07_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("../originals/h_train/457_z/173/14_32_27_31_07_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("../originals/heights/157/16_13_54_19_07_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("../originals/heights/157/16_13_54_19_07_2024_IMG_RIGHT.jpg")

    # img_left = cv2.imread("../originals/prof_alturas/300_z/15_00_39_19_08_2024_IMG_LEFT.jpg")
    # img_right = cv2.imread("../originals/prof_alturas/300_z/15_00_39_19_08_2024_IMG_RIGHT.jpg")
    
    img_left = cv2.imread("../originals/distances/300/14_06_13_13_05_2024_IMG_LEFT.jpg")
    img_right = cv2.imread("../originals/distances/300/14_06_13_13_05_2024_IMG_RIGHT.jpg")
    
    if img_left is None or img_right is None:
        raise FileNotFoundError("Una o ambas imágenes no pudieron ser cargadas. Verifique las rutas.")


    
    # Cargar configuración desde el archivo JSON
    config = load_config("profiles/MATLAB.json")
    

    img_left, img_right = rectify_images(img_left, img_right, config=config['profile_name'])
    # Asumiendo que queremos usar el método SGBM, ajusta si es WLS-SGBM, RAFT o SELECTIVE según tu configuración
    method = 'RAFT'

    # convert_to_gray("./raft_demo_output/output_1.png","./raft_demo_output/gray_output_1.png")
    # convert_to_gray("./seletive_demo_output/output_1.png","./seletive_demo_output/gray_output_1.png")
    # #TEST MAPA DISPARIDAD
    #test_disparity_map(img_left, img_right, config, method)

    # #TEST NUBE DE PUNTOS DENSA
    #test_point_cloud(img_left, img_right, config, method, use_max_disparity=True, normalized=False)


    # #TEST NUBE DE PUNTOS NO DENSA TOTAL
    #test_filtered_point_cloud(img_left, img_right, config, method, use_roi=True, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA TOTAL
    # test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)


    # #TEST NUBE DE PUNTOS NO DENSA INDIVIDUAL
    #test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi=True, use_max_disparity=True, normalized=False)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA INDIVIDUAL
    #test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)


    # # #TEST CALCULO DE ALTURAS
    # test_estimate_height_from_point_cloud(img_left, img_right, config, method, use_roi=False, use_max_disparity=True, normalized=True)
    # print("")
    #points, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, False, True)
    #points, colors = generate_dense_point_cloud(img_left, img_right, config, method, True, True)
    # Seleccionar un subconjunto aleatorio de puntos, incluyendo el origen
    # NUM_POINTS_TO_DRAW = 500000
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
    # print("TESTING IS ENDING...")


    #TEST FEATURE EXTRACTION
    # test_filtered_point_cloud_with_features(img_left, img_right, config, method="RAFT", use_roi=False, use_max_disparity=True, normalized=True)

    #TEST HEIGHT FROM FACE
    test_estimate_height_from_face_proportions(img_left, img_right, config)
    