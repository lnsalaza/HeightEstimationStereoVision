import json
import cv2

import numpy as np
import dense_point_cloud.pc_generation as pcGen
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
from dense_point_cloud.point_cloud import compute_centroid, compute_disparity, generate_dense_point_cloud, generate_combined_filtered_point_cloud, generate_individual_filtered_point_clouds, compute_centroids 

######################### NEXT FUNCTION ARE JUST FOR TESTING PURPOSES #################################
def test_disparity_map(img_left, img_right, config, method):
    # Calcular el mapa de disparidad
    disparity_map = compute_disparity(img_left, img_right, config, method)


    # Visualizar el mapa de disparidad generado
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

def test_point_cloud(img_left, img_right, config, method, use_max_disparity):
    # Generar la nube de puntos 3D
    point_cloud, colors = generate_dense_point_cloud(img_left, img_right, config, method, use_max_disparity)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/densaDEMO")
    # Convertir los datos de la nube de puntos y colores a formato Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar los colores a [0, 1]

    # Crear una ventana de visualización
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="3D Point Cloud", width=800, height=600)

    # Añadir la nube de puntos a la ventana de visualización
    viewer.add_geometry(pcd)

    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 1  # Establecer el tamaño de los puntos

    # Ejecutar la visualización
    viewer.run()
    viewer.destroy_window()

def test_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)

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

def test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar la nube de puntos 3D filtrada y combinada
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)

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

def test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity)
    
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


def test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi, use_max_disparity):
    # Generar listas de nubes de puntos y colores para cada objeto detectado
    point_cloud_list, color_list = generate_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi, use_max_disparity)
    
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




def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

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
    # test_point_cloud(img_left, img_right, config, method, use_max_disparity=False)


    # #TEST NUBE DE PUNTOS NO DENSA TOTAL
    #test_filtered_point_cloud(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA TOTAL
    # test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)



    # #TEST NUBE DE PUNTOS NO DENSA INDIVIDUAL
    # test_individual_filtered_point_clouds(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    # #TEST CENTROIDE EN NUBE DE PUNTOS NO DENSA INDIVIDUAL
    # test_individual_filtered_point_cloud_with_centroid(img_left, img_right, config, method, use_roi=False, use_max_disparity=True)

    points, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, False, True)
    # Seleccionar un subconjunto aleatorio de puntos, incluyendo el origen
    NUM_POINTS_TO_DRAW = len(points)
    subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW - 1,), replace=False)
    subset = np.append(subset, points.shape[0] - 1)  # Asegurar que el origen esté incluido
    points_subset = points[subset]
    colors_subset = colors[subset]

    x, y, z = points_subset.T

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z, # flipped to make visualization nicer
                mode='markers',
                marker=dict(size=1, color=colors_subset)
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
            )
        )
    )
    fig.show()
    print("hola")


