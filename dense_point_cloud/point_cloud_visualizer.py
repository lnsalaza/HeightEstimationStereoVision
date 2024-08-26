import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


# def visualize_sparse_point_cloud(pcd_file, eps=100, min_samples=5000):

#     # Leer la nube de puntos
#     pcd = o3d.io.read_point_cloud(pcd_file)
#     points = np.asarray(pcd.points)
#     print(points.size)
#     colors = np.array([[0, 0, 1] for i in range(len(pcd.points))])

# # Asignar los colores a la nube de puntos
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Aplicar DBSCAN
#     db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
#     labels = db.labels_

#     # Encontrar el número de clusters (excluyendo ruido)
#     unique_labels = set(labels)
#     unique_labels.discard(-1)  # Eliminar el ruido (-1)


#      # Obtener el Axis-Aligned Bounding Box (AABB)
#     aabb = pcd.get_axis_aligned_bounding_box()
#     aabb.color = (1, 0, 0)

#     # Crear visualización de clusters y centroides
#     geometries = [pcd, aabb]
#     origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
#     geometries.append(origin)

#     for label in unique_labels:
#         cluster_points = points[labels == label]
#         centroid = cluster_points.mean(axis=0)
        
#         # Crear una pequeña esfera en el centroide para visualizarlo
#         centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
#         centroid_sphere.translate(centroid)
#         centroid_sphere.paint_uniform_color([1, 0, 0])  # Color rojo para el centroide
#         geometries.append(centroid_sphere)

#     # Crear el visualizador
#     viewer = o3d.visualization.Visualizer()
#     viewer.create_window()
    
#     # Visualizar las geometrías
#     o3d.visualization.draw_geometries(geometries)

#     # Destruir la ventana del visualizador
#     viewer.destroy_window()



# Clase encargada de la normalizacion estandar de las nubes de puntos

class PointCloudScaler:
    def __init__(self, reference_point, scale_factor):
        self.reference_point = np.array(reference_point)
        self.scale_factor = scale_factor

    def calculate_scaled_positions(self, points):
        # Restar el punto de referencia a todos los puntos
        shifted_points = points - self.reference_point
        
        # Escalar los puntos
        scaled_points = self.scale_factor * shifted_points
        
        # Volver a mover los puntos al sistema de referencia original
        new_positions = scaled_points + self.reference_point
        
        return new_positions

    def scale_cloud(self, points):
        # Procesa todos los puntos sin dividirlos en trozos ni usar procesamiento paralelo
        new_positions = self.calculate_scaled_positions(points)
        return new_positions

def correct_depth_o3d(points, alpha=0.5):
    """
    Aplica una corrección de profundidad a una nube de puntos 3D numpy array.
    
    :param points: Numpy array de puntos 3D
    :param alpha: Parámetro de la transformación de potencia (0 < alpha < 1)
    :return: Numpy array de puntos 3D corregidos
    """
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    Z_safe = np.where(Z == 0, np.finfo(float).eps, Z)
    Z_corrected = Z_safe ** alpha
    X_corrected = X * (Z_corrected / Z_safe)
    Y_corrected = Y * (Z_corrected / Z_safe)
    corrected_points = np.vstack((X_corrected, Y_corrected, (0.6947802265318861*Z_corrected) + -14.393348239171985)).T
    return corrected_points

def process_numpy_point_cloud(points_np, reference_point=[0, 0, 0], scale_factor=1, alpha=1):
    # Escalar la nube de puntos
    scaler = PointCloudScaler(reference_point=reference_point, scale_factor=scale_factor)
    scaled_points_np = scaler.scale_cloud(points_np)
    
    # Aplicar corrección de profundidad
    corrected_points_np = correct_depth_o3d(scaled_points_np, alpha)
    
    return scaled_points_np

def visualize_sparse_point_cloud(pcd_file, eps=500, min_samples=10):
    # Leer la nube de puntos
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    print(points.shape[0])
    colors = np.array([[0, 0, 1] for i in range(len(pcd.points))])

    # Asignar los colores a la nube de puntos
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Aplicar DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    # Encontrar el número de clusters (excluyendo ruido)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Eliminar el ruido (-1)

    # Obtener el Axis-Aligned Bounding Box (AABB)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)

    # Crear visualización de clusters y centroides
    geometries = [pcd, aabb]
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    # geometries.append(origin)

    for label in unique_labels:
        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)

        # Crear una pequeña esfera en el centroide para visualizarlo
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.5)
        centroid_sphere.translate(centroid)
        centroid_sphere.paint_uniform_color([1, 0, 0])  # Color rojo para el centroide
        geometries.append(centroid_sphere)

    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    # Ajustar las opciones de renderización
    # opt = viewer.get_render_option()
    # opt.point_size = 1  # Tamaño de los puntos

    # Añadir geometrías al visualizador
    for geometry in geometries:
        viewer.add_geometry(geometry)
    
    # Iniciar el visualizador
    viewer.run()

    # Destruir la ventana del visualizador
    viewer.destroy_window()

def create_mark_lines(width=200, height=40, depth=1000, interval=100):


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


def visualize_dense_point_cloud(pcd_file):
    # Crear las líneas de referencia
    # mark_lines = create_mark_lines(width=1600, height=600, depth=3000, interval=100)
    mark_lines = create_mark_lines(width=int(1600*0.280005), height=int(600*0.280005), depth=int(3000*0.280005), interval=int(100*0.280005))
    # Leer la nube de puntos densa
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Convertir a un array numpy para procesamiento
    
    points_np = np.asarray(pcd.points)
    processed_points_np = process_numpy_point_cloud(points_np)
    # Procesar la nube de puntos numpy


    # Crear una nueva nube de puntos Open3D a partir del array numpy procesado
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(processed_points_np)

    # Definir el rango de Z para el recorte (opcional, no utilizado actualmente)
    z_min, z_max = 0, 1000000
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-float('inf'), -float('inf'), z_min), max_bound=(float('inf'), float('inf'), z_max))
    
    # Crear el marco de coordenadas en el origen
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])

    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    
    # Agregar las geometrías al visualizador
    # viewer.add_geometry(pcd)  # Nube de puntos original
    viewer.add_geometry(processed_pcd)  # Nube de puntos procesada
    viewer.add_geometry(origin)  # Marco de coordenadas
    # viewer.add_geometry(mark_lines)  # Líneas de referencia
    
    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 1
    
    # Ejecutar el visualizador
    viewer.run()
    
    # Destruir la ventana del visualizador
    viewer.destroy_window()


def print_centroid_z_coordinates(centroid_file):

    centroid_cloud = o3d.io.read_point_cloud(centroid_file)
    centroids = np.asarray(centroid_cloud.points)
    z_coordinates = centroids[:, 2]
    print("Coordenadas Z de los centroides:")
    for z in z_coordinates:
        print(z)

# Uso de las funciones
if __name__ == "__main__":

    # config = "matlab_1"
    config = "SGBM"
    mask = "keypoints"
    situacion = "150_A"
    
    # filepath = f"../point_clouds/{config}/{mask}_disparity/{config}_{situacion}"
    filepath = "../../tmp/point_clouds/intermediate_point_cloud"
    # filepath = "../point_clouds/SGBM/keypoints_disparity/SGBM_150_A"
    # filepath = "../point_clouds/DEMO/densaDEMO"
    # Visualización de la nube de puntos densa
    dense_pcd_file = f"{filepath}.ply"

    visualize_dense_point_cloud(dense_pcd_file)

    
    # Visualización de la nube de puntos dispersa
    # sparse_pcd_file = f"{filepath}_filtered_dense.ply"
    # centroid_file = f"{filepath}_person0_centroids.ply"

    # Imprimir coordenadas Z de los centroides
    # print_centroid_z_coordinates(centroid_file)
    
    # Visualizar la nube de puntos dispersa
    # visualize_sparse_point_cloud(sparse_pcd_file)

    



