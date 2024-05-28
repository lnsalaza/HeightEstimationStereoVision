import open3d as o3d
import numpy as np
def visualize_sparse_point_cloud(pcd_file, centroid_file):

    # Leer la nube de puntos y los centroides
    pcd = o3d.io.read_point_cloud(pcd_file)
    centroid_cloud = o3d.io.read_point_cloud(centroid_file)
    
    # Obtener el Axis-Aligned Bounding Box (AABB)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    
    # Obtener el Oriented Bounding Box (OBB)
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0, 1, 0)
    
    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    
    # Visualizar las geometrías
    o3d.visualization.draw_geometries([pcd, centroid_cloud, aabb, obb])
    
    # Destruir la ventana del visualizador
    viewer.destroy_window()

def visualize_dense_point_cloud(pcd_file):

    # Leer la nube de puntos densa
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # Crear el visualizador
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    
    # Agregar la geometría
    viewer.add_geometry(pcd)
    
    # Configurar opciones de renderizado
    opt = viewer.get_render_option()
    opt.point_size = 0.1
    
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
    # Visualización de la nube de puntos dispersa
    sparse_pcd_file = "./point_clouds/new_calibration_200_front_original.ply"
    centroid_file = "./point_clouds/new_calibration_200_front_centroids.ply"
    
    # Imprimir coordenadas Z de los centroides
    print_centroid_z_coordinates(centroid_file)
    
    # Visualizar la nube de puntos dispersa
    visualize_sparse_point_cloud(sparse_pcd_file, centroid_file)
    
    # Visualización de la nube de puntos densa
    dense_pcd_file = "./point_clouds/new_200_front_dense.ply"
    visualize_dense_point_cloud(dense_pcd_file)