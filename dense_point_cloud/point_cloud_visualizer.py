import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
def visualize_sparse_point_cloud(pcd_file, eps=100, min_samples=6):

    # Leer la nube de puntos
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
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
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    geometries.append(origin)

    for label in unique_labels:
        cluster_points = points[labels == label]
        centroid = cluster_points.mean(axis=0)
        
        # Crear una pequeña esfera en el centroide para visualizarlo
        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        centroid_sphere.translate(centroid)
        centroid_sphere.paint_uniform_color([1, 0, 0])  # Color rojo para el centroide
        geometries.append(centroid_sphere)

    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()

    # Visualizar las geometrías
    o3d.visualization.draw_geometries(geometries)

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
    
    
    mark_lines = create_mark_lines()
    # Leer la nube de puntos densa
    pcd = o3d.io.read_point_cloud(pcd_file)
    # # ESCALADO BETA
    # scale_factor = 1.0
    # scaling_matrix = np.eye(4)
    # scaling_matrix[:3, :3] *= scale_factor

    # pcd = pcd.transform(scaling_matrix)

    z_min, z_max = 0, 1000000
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-float('inf'), -float('inf'), z_min), max_bound=(float('inf'), float('inf'), z_max))
    cropped_pcd = pcd.crop(bounding_box)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0,0,0])

    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    
    # Agregar la geometría
    #viewer.add_geometry(cropped_pcd)
    viewer.add_geometry(pcd)
    viewer.add_geometry(origin)
    # viewer.add_geometry(mark_lines)
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

    config = "matlab_1"
    mask = "roi"
    situacion = "150_a"

    filepath = f"./point_clouds/{config}/{mask}_disparity/{config}_{situacion}"

    # Visualización de la nube de puntos densa
    dense_pcd_file = f"{filepath}_dense.ply"
    visualize_dense_point_cloud(dense_pcd_file)

    
    # Visualización de la nube de puntos dispersa
    sparse_pcd_file = f"{filepath}_person1_original.ply"
    # centroid_file = f"{filepath}_person0_centroids.ply"

    # Imprimir coordenadas Z de los centroides
    # print_centroid_z_coordinates(centroid_file)
    
    # Visualizar la nube de puntos dispersa
    visualize_dense_point_cloud(sparse_pcd_file)

    



