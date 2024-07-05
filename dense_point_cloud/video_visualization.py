import time
import numpy as np
import open3d as o3d
import os
import re

def listar_archivos(carpeta, extension, pista_nombre=''):
    """
    Recorrer los archivos en una carpeta que cumplan con la extensión y pista de nombre especificadas.

    :param carpeta: Ruta a la carpeta que contiene los archivos.
    :param extension: Extensión de los archivos a buscar (por ejemplo, ".ply").
    :param pista_nombre: Pista del nombre del archivo (opcional).
    :return: Lista de archivos que cumplen con las características especificadas.
    """
    archivos_cumplen = []
    for archivo in os.listdir(carpeta):
        if (archivo.endswith(extension) and (pista_nombre in archivo)):
            archivos_cumplen.append(archivo)
    return archivos_cumplen

# Configurar carpeta y criterios de archivo
carpeta = '../dense_point_cloud/point_clouds/video/L/matlab_1/keypoint_disparity'
extension = '.ply'
pista_nombre = 'dense'  # Dejar vacío si no se necesita

# Obtener y ordenar archivos
archivos = listar_archivos(carpeta, extension, pista_nombre)
archivos = sorted(archivos, key=lambda s: int(re.findall(r'\d+', s)[1]))


view_config = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 219.53413391113281, 58.269557952880859, 850.05814117438558 ],
			"boundingbox_min" : [ -90.282096862792969, -116.62229156494141, -0.59999999999999998 ],
			"field_of_view" : 60.0,
			"front" : [ -0.095835554820780974, 0.019926798296555224, -0.99519770354530135 ],
			"lookat" : [ 30.82705712235359, -8.267747845649291, 440.96600777414471 ],
			"up" : [ -0.041756373084997557, -0.99899999246766535, -0.015981875872099108 ],
			"zoom" : 0.19999999999999998
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


# Crear un objeto de visualización y ventana
vis = o3d.visualization.Visualizer()
vis.create_window()

print(vis.get_view_control())

vis.get_view_control().set_front(view_config["trajectory"][0]["front"])
vis.get_view_control().set_lookat(view_config["trajectory"][0]["lookat"])
vis.get_view_control().set_up(view_config["trajectory"][0]["up"])
vis.get_view_control().set_zoom(view_config["trajectory"][0]["zoom"])

opt = vis.get_render_option()
opt.point_size = 1

# Crear objeto de nube de puntos
point_cloud_o3d = o3d.geometry.PointCloud()

start_time = time.time()
prev_sec = -1
translation_index = 0
translation_speed = 0.01
first_call = True
counter = 0
# Iterar sobre los archivos y mostrarlos en secuencia
for archivo in archivos:
    office = o3d.io.read_point_cloud(os.path.join(carpeta, archivo))
    points = np.asarray(office.points) + translation_index * translation_speed
    

    # Modificar la nube de puntos original para mostrar una secuencia
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.colors = office.colors

    # Añadir o actualizar objetos geométricos
    if first_call:
        vis.add_geometry(point_cloud_o3d, )  # Añadir la nube de puntos
        first_call = False  # Cambiar la bandera
    else:
        vis.update_geometry(point_cloud_o3d)  # Actualizar la nube de puntos

    translation_index += 1
    
    # Actualizar la ventana de visualización
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"{carpeta}/temp_{counter}.jpg")
    counter +=1
    time.sleep(1)  # Esperar 1 segundo antes de mostrar la siguiente nube de puntos

# Mantener la ventana abierta hasta que el usuario la cierre

vis.run()
vis.destroy_window()