import open3d as o3d
import numpy as np
import csv
import os
from typing import List, Tuple
from pathlib import Path
import shutil
import zipfile

def create_point_cloud(points, colors=None):
    """
    Crea un objeto PointCloud de Open3D a partir de arrays numpy de puntos y colores.

    Args:
        points (np.array): Array numpy de forma (N, 3) con las coordenadas XYZ de los puntos.
        colors (np.array): Array numpy de forma (N, 3) con los colores RGB de los puntos, normalizados entre 0 y 255.

    Returns:
        o3d.geometry.PointCloud: Objeto PointCloud creado.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalizar colores a [0, 1]
    return pcd

def prepare_point_cloud(cloud: np.array, colors: np.array = None, temporary_storage_path: str = "../tmp/point_clouds/intermediate_point_cloud.ply"):
    """
    Prepara y guarda temporalmente una nube de puntos en un formato intermedio estándar (PLY), asegurándose de limpiar cualquier archivo existente.

    Args:
        cloud (np.array): La nube de puntos a guardar como un array numpy.
        colors (np.array): Array numpy de colores RGB (opcional).
        temporary_storage_path (str): Ruta completa del archivo donde se guardará temporalmente la nube de puntos.

    Returns:
        Tuple[bool, str]: True y el path del archivo si se guarda correctamente, False y un mensaje de error en caso contrario.
    """
    try:
        
        directory = os.path.dirname(temporary_storage_path)

         # Limpiar el directorio completo 
        if os.path.exists(directory):
            shutil.rmtree(directory)

        # Asegurarse de que el directorio de salida exista vacio
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Directorio {directory} creado.")

        

        # Crear la nube de puntos utilizando la función create_point_cloud
        point_cloud = create_point_cloud(cloud, colors)

        # Guardar la nube de puntos en formato PLY como estándar intermedio
        o3d.io.write_point_cloud(temporary_storage_path, point_cloud)

        return True, temporary_storage_path
    except Exception as e:
        return False, f"Error al guardar el archivo intermedio: {str(e)}"


def prepare_individual_point_clouds(
    clouds: List[np.array],
    colors_list: List[np.array],
    keypoints3d_list: List[List[List[float]]],
    base_path: str = "../tmp/point_clouds_filtered/"
) -> Tuple[bool, str]:
    """
    Prepara y guarda temporalmente múltiples nubes de puntos y sus correspondientes keypoints 3D, cada uno en su propio archivo.

    Args:
        clouds (List[np.array]): Lista de arrays numpy de nubes de puntos a guardar.
        colors_list (List[np.array]): Lista de arrays numpy de colores RGB correspondientes.
        keypoints3d_list (List[List[List[float]]]): Lista de listas de keypoints 3D para cada nube de puntos.
        base_path (str): Ruta base para almacenar los archivos temporales.

    Returns:
        Tuple[bool, str]: True y la ruta base de los archivos si se guardan correctamente, False y un mensaje de error en caso contrario.
    """
    try:
        # Asegurarse de que el directorio de salida exista y esté limpio
        if os.path.exists(base_path):
            shutil.rmtree(base_path)
        os.makedirs(base_path, exist_ok=True)

        # Guardar cada nube de puntos y keypoints en archivos individuales
        for idx, (cloud, colors, keypoints3d) in enumerate(zip(clouds, colors_list, keypoints3d_list)):
            cloud_path = os.path.join(base_path, f"cloud_{idx}.ply")
            keypoints_path = os.path.join(base_path, f"keypoints_{idx}.csv")

            # Crear la nube de puntos utilizando la función create_point_cloud
            point_cloud = create_point_cloud(cloud, colors)

            # Guardar la nube de puntos en formato PLY
            o3d.io.write_point_cloud(cloud_path, point_cloud)

            # Guardar keypoints en CSV
            with open(keypoints_path, 'w', newline='') as file:
                writer = csv.writer(file)
                for kp in keypoints3d:
                    writer.writerow(kp)

        return True, base_path
    except Exception as e:
        return False, f"Error al preparar los archivos de nubes de puntos: {str(e)}"
    


def convert_point_cloud_format(output_format: str, file_path: str = "../tmp/point_clouds/intermediate_point_cloud.ply") -> Tuple[bool, str]:
    """
    Verifica la presencia de un archivo de nube de puntos en un path específico y lo convierte al formato solicitado.

    Args:
        output_format (str): Formato al que se desea convertir la nube de puntos.
        file_path (str): Ruta del archivo temporal de la nube de puntos, con valor predeterminado.

    Returns:
        Tuple[bool, str]: True y la ruta del archivo convertido si la operación es exitosa, False y un mensaje de error en caso contrario.
    """
    supported_formats = ['ply', 'xyz', 'pcd', 'pts', 'xyzrgb']
    if output_format not in supported_formats:
        return False, f"Formato '{output_format}' no soportado. Elija entre {supported_formats}."

    if not os.path.exists(file_path):
        return False, "Archivo de nube de puntos no encontrado."

    try:
        # Ruta de salida para el archivo convertido
        output_path = file_path.replace(".ply", f".{output_format}")
        
        
        cloud = o3d.io.read_point_cloud(file_path)

        # Guardar la nube de puntos en el nuevo formato
        if output_format == 'ply':
            o3d.io.write_point_cloud(output_path, cloud)
        elif output_format == 'xyz':
            np.savetxt(output_path, np.asarray(cloud.points), fmt='%f %f %f')
        elif output_format == 'pcd':
            o3d.io.write_point_cloud(output_path, cloud, write_ascii=False)
        elif output_format == 'pts':
            np.savetxt(output_path, np.asarray(cloud.points), fmt='%f %f %f')
        elif output_format == 'xyzrgb':
            o3d.io.write_point_cloud(output_path, cloud)
        else:
            return False, f"Formato de salida '{output_format}' no soportado."

        return True, output_path
    except Exception as e:
        return False, f"Error al convertir el archivo: {str(e)}"
    

def convert_individual_point_clouds_format(
    output_format: str,
    base_path: str = "../tmp/point_clouds_filtered/",
    output_zip_path: str = "../tmp/point_clouds_filtered/converted_point_clouds.zip"
) -> Tuple[bool, str]:
    """
    Convierte las nubes de puntos individuales a un formato especificado y las comprime junto con los keypoints en un archivo ZIP.

    Args:
        output_format (str): Formato al que se desea convertir las nubes de puntos ('ply', 'xyz', 'pcd', 'pts', 'xyzrgb').
        base_path (str): Ruta base donde se encuentran las nubes de puntos y keypoints guardados temporalmente.
        output_zip_path (str): Ruta del archivo ZIP final que se generará.

    Returns:
        Tuple[bool, str]: True y la ruta del archivo ZIP si la operación es exitosa, False y un mensaje de error en caso contrario.
    """
    supported_formats = ['ply', 'xyz', 'pcd', 'pts', 'xyzrgb']
    if output_format not in supported_formats:
        return False, f"Formato '{output_format}' no soportado. Elija entre {supported_formats}."

    try:
        # Asegurarse de que no exista un archivo ZIP previo
        if os.path.exists(output_zip_path):
            os.remove(output_zip_path)

        # Crear un nuevo archivo ZIP
        with zipfile.ZipFile(output_zip_path, 'w') as zipf:
            for file_name in os.listdir(base_path):
                if file_name.endswith(".ply"):
                    file_path = os.path.join(base_path, file_name)

                    # Leer la nube de puntos
                    cloud = o3d.io.read_point_cloud(file_path)

                    # Definir el nuevo nombre del archivo
                    converted_file_path = file_path.replace(".ply", f".{output_format}")

                    # Convertir y guardar la nube de puntos en el formato solicitado
                    if output_format == 'ply':
                        o3d.io.write_point_cloud(converted_file_path, cloud)
                    elif output_format == 'xyz':
                        np.savetxt(converted_file_path, np.asarray(cloud.points), fmt='%f %f %f')
                    elif output_format == 'pcd':
                        o3d.io.write_point_cloud(converted_file_path, cloud, write_ascii=False)
                    elif output_format == 'pts':
                        np.savetxt(converted_file_path, np.asarray(cloud.points), fmt='%f %f %f')
                    elif output_format == 'xyzrgb':
                        o3d.io.write_point_cloud(converted_file_path, cloud)

                    # Agregar la nube de puntos convertida al archivo ZIP
                    zipf.write(converted_file_path, os.path.basename(converted_file_path))

                elif file_name.endswith(".csv"):
                    keypoints_path = os.path.join(base_path, file_name)
                    zipf.write(keypoints_path, os.path.basename(keypoints_path))

        return True, output_zip_path
    except Exception as e:
        return False, f"Error al convertir y comprimir los archivos: {str(e)}"