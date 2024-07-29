import os
import re
import cv2
import csv
import string
import joblib
import numpy as np
import pc_generation as pcGen
import pc_generation_ML as pcGen_ML
import matplotlib.pyplot as plt


from bridge_selective import get_SELECTIVE_disparity_map
from bridge_raft import get_RAFT_disparity_map
# Definición de los videos y matrices de configuración
configs = {
    'matlab_1': {
        'LEFT_VIDEO': '../videos/rectified/matlab_1/distance_left.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_1/distance_right.avi',
        # 'LEFT_VIDEO': '../videos/rectified/matlab_1/16_35_42_26_02_2024_VID_LEFT.avi',
        # 'RIGHT_VIDEO': '../videos/rectified/matlab_1/16_35_42_26_02_2024_VID_RIGHT.avi',
        'MATRIX_Q': '../config_files/matlab_1/newStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "../datasets/models/matlab_1/LASER2.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'opencv_1': {
        'LEFT_VIDEO': '../videos/rectified/opencv_1/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/opencv_1/right_rectified.avi',
        'MATRIX_Q': '../config_files/opencv_1/stereoMap.xml',
        'disparity_to_depth_map': 'disparityToDepthMap',
        'model': "../datasets/models/z_estimation_opencv_1_keypoint_ln_model.pkl",
        'numDisparities': 52,
        'blockSize': 10, 
        'minDisparity': 0,
        'disp12MaxDiff': 36,
        'uniquenessRatio': 39,
        'preFilterCap': 25,
        'mode': cv2.StereoSGBM_MODE_HH
    },
    'matlab_2': {
        'LEFT_VIDEO': '../videos/rectified/matlab_2/left_rectified.avi',
        'RIGHT_VIDEO': '../videos/rectified/matlab_2/right_rectified.avi',
        'MATRIX_Q': '../config_files/laser_config/including_Y_rotation_random/iyrrStereoMap.xml',
        'disparity_to_depth_map': 'disparity2depth_matrix',
        'model': "../datasets/models/z_estimation_matlab_1_keypoint_ln_model_LASER.pkl",
        'numDisparities': 68,
        'blockSize': 7, 
        'minDisparity': 5,
        'disp12MaxDiff': 33,
        'uniquenessRatio': 10,
        'preFilterCap': 33,
        'mode': cv2.StereoSGBM_MODE_HH
    }
}

situations = {
    '150_front': 60,
    '150_bodyside_variant': 150,
    '150_500': 3930,
    '200_front': 720,
    '200_shaking_hands_variant': 750,
    '200_400_front': 4080,
    '200_400_sitdown': 6150,
    '200_400_sitdown_side_variant': 6240,
    '250_front': 1020,
    '250_oneback_one_front_variant': 1050,
    '250_side_variant': 1140,
    '250_350': 4200,
    '250_500': 6900,
    '250_600': 4470,
    '250_600_perspective_variant': 4590,
    '300_front': 1290,
    '350_front': 1530,
    '350_side_variant': 1800,
    '400_front': 2010,
    '400_oneside_variant': 2130,
    '400_120cm_h_variant': 5160,
    '450_front': 2310,
    '450_side_variant': 2370,
    '450_600': 4710,
    '500_front': 2700,
    '500_oneside_variant': 2670,
    '550_front': 3000,
    '550_oneside_variant': 2940,
    '600_front': 3240,
    '600_oneside_variant': 3150
}

# situations= {
#     "L_0": 250,
#     "L_1": 260,
#     "L_2": 270,
#     "L_3": 280,
#     "L_4": 290,
#     "L_5": 300,
#     "L_6": 310,
#     "L_7": 320,
#     "L_8": 330,
#     "L_9": 340,
#     "L_10": 350,
#     "L_11": 360,
#     "L_12": 370
#}
# situations= {
#     "I_0":1000,
#     "I_1":1010,
#     "I_2":1020,
#     "I_3":1030,
#     "I_4":1040,
#     "I_5":1050,
#     "I_6":1060,
#     "I_7":1070,
#     "I_8":1080,
#     "I_9":1090,
#     "I_10":1100,
#     "I_11":1110,
#     "I_12":1120,
# }

# Función para seleccionar configuración de cámara
def select_camera_config(camera_type):
    config = configs[camera_type]
    LEFT_VIDEO = config['LEFT_VIDEO']
    RIGHT_VIDEO = config['RIGHT_VIDEO']
    MATRIX_Q = config['MATRIX_Q']
    
    fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
    Q = fs.getNode(config['disparity_to_depth_map']).mat()
    fs.release()
    
    return LEFT_VIDEO, RIGHT_VIDEO, Q

def extract_frame(video_path, n_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    retval, frame = cap.read()
    cap.release()
    if not retval:
        raise ValueError(f"No se pudo leer el frame {n_frame}")
    return frame

# Función para extraer un frame específico de los videos izquierdo y derecho
def extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color=True, save=True):
    image_l = extract_frame(LEFT_VIDEO, n_frame)
    image_r = extract_frame(RIGHT_VIDEO, n_frame)

    if not color:
        image_l = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

    if save:
        cv2.imwrite(f"../images/image_l_{n_frame}.png", image_l)
        cv2.imwrite(f"../images/image_r_{n_frame}.png", image_r)
    
    return image_l, image_r
    

# Función para extraer frames según la situación y configuración de cámara
def extract_situation_frames(camera_type, situation, color=True, save=True):
    if situation in situations:
        n_frame = situations[situation]
        LEFT_VIDEO, RIGHT_VIDEO, Q = select_camera_config(camera_type)
        return extract_image_frame(LEFT_VIDEO, RIGHT_VIDEO, n_frame, color, save), Q
    else:
        raise ValueError("Situación no encontrada en el diccionario.")

# Función para la creacion de un filtro logico en el centroide
def filter_points_by_optimal_range(point_cloud, centroid, m_initial=30):
    z_centroid = centroid[2]
    m = m_initial  # Puedes ajustar esto si necesitas una relación más compleja
    lower_bound = z_centroid - m
    upper_bound = z_centroid + m

    # Crear una máscara lógica para filtrar los puntos en el rango óptimo
    mask = (point_cloud[:, 2] >= lower_bound) & (point_cloud[:, 2] <= upper_bound)
    filtered_points = point_cloud[mask]

    return filtered_points

def get_Y_bounds(filtered_points):
    if filtered_points.size == 0:
        return None, None

    y_min = np.min(filtered_points[:, 1])
    y_max = np.max(filtered_points[:, 1])

    return y_min, y_max

def read_image_pairs_by_distance(base_folder):
    image_pairs_by_distance = {}

    # Recorre todas las subcarpetas en la carpeta base
    for subdir, dirs, files in os.walk(base_folder):
        # Extrae la distancia (nombre de la subcarpeta)
        distance = os.path.basename(subdir)
        
        if subdir != base_folder:
            # if distance not in image_pairs_by_distance:
            image_pairs_by_distance[distance] = []

            # Filtra las imágenes LEFT y RIGHT
            left_images = sorted([f for f in files if 'IMG_LEFT' in f])
            right_images = sorted([f for f in files if 'IMG_RIGHT' in f])

            # Empareja las imágenes por su timestamp
            for left_img in left_images:
                timestamp = left_img.split('_IMG_LEFT')[0]
                corresponding_right_img = timestamp + '_IMG_RIGHT.jpg'
                if corresponding_right_img in right_images:
                    left_img_path = os.path.join(subdir, left_img)
                    right_img_path = os.path.join(subdir, corresponding_right_img)
                    
                    # # Lee las imágenes con OpenCV
                    # img_left = cv2.imread(left_img_path)
                    # img_right = cv2.imread(right_img_path)
                    
                    if left_img_path is not None and right_img_path is not None:
                        image_pairs_by_distance[distance].append((left_img_path, right_img_path))
                    else:
                        print(f"Error al leer las imágenes: {left_img_path} o {right_img_path}")
    
    return image_pairs_by_distance

def graficar_alturas(alturas_estimadas, altura_minima, altura_maxima):
    """
    Función para graficar alturas estimadas.

    :param alturas_estimadas: Lista de alturas estimadas.
    :param altura_minima: Valor mínimo del rango de altura.
    :param altura_maxima: Valor máximo del rango de altura.
    """
    # Creación de la figura y los ejes
    fig, ax = plt.subplots()

    # Gráfico de la línea de alturas estimadas
    ax.plot(alturas_estimadas, marker='o', linestyle='-', color='b')

    # Configuración de los límites de los ejes
    ax.set_xlim(0, len(alturas_estimadas) - 1)
    ax.set_ylim(altura_minima, altura_maxima)

    # Etiquetas de los ejes
    ax.set_xlabel('Índice de la medición')
    ax.set_ylabel('Altura estimada (cm)')

    # Título de la gráfica
    ax.set_title('Comportamiento de las alturas estimadas')

    plt.savefig("IMG_alturas")
    plt.close()

def swap_numbers_in_situation(situation):
    # Usar regex para extraer números de la situación
    numbers = re.findall(r'\d+', situation)
    numbers = [int(num) for num in numbers]

    if len(numbers) == 2:
        return situation.replace(str(numbers[0]), "{temp}").replace(str(numbers[1]), str(numbers[0])).replace("{temp}", str(numbers[1]))
    else:
        return situation

def get_adjusted_situation(situation, data):
    # Verificar si la situación ya está en el diccionario
    existing_entry = next((entry for entry in data if entry["situation"].startswith(situation)), None)
    if existing_entry:
        # Intercambiar los números en la situación
        return swap_numbers_in_situation(situation)
    else:
        return situation


# Flujo principal para todas las situaciones
data = []
data_height = []
camera_type = 'matlab_1'
mask_type = 'roi'
is_roi = (mask_type == "roi")
situation = "450_600"
model_path = configs[camera_type]['model']
alphabet = string.ascii_lowercase
# Cargar el modelo de regresión lineal entrenado
model = joblib.load(model_path)
method_used = "RAFT" #OPTIONS: "SGBM". "RAFT", "SELECTIVE"


print(f"{method_used} ESTA SIENDO USADO")
fx, fy, cx1, cy = 1429.4995220185822, 1430.4111785502332, 929.8227256572083, 506.4722541384677
cx2 = 936.8035788332203
baseline = 32.95550620237698 # in millimeters


# Definir si se debe aplicar la corrección de la nube de puntos
apply_correction = False


################################################################################################################################

# for situation in situations:
#     try:
#         print(f"\nProcesando situación: {situation}")
#         (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
#         img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
#         img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
        
#         disparity = pcGen.compute_disparity(img_l, img_r, configs[camera_type])

#         # # Generar nube de puntos densa sin filtrado adicional
#         dense_point_cloud, dense_colors = pcGen.disparity_to_pointcloud(disparity, Q, img_l)
#         dense_point_cloud = dense_point_cloud.astype(np.float64)

#         # Corrección de nube densa 
#         # dense_point_cloud = pcGen.point_cloud_correction(dense_point_cloud, model)

#         base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}_h_train"
#         pcGen.save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

#         # # Generar nube de puntos con filtrado y aplicar DBSCAN
#         # point_cloud, colors, eps, min_samples = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type, use_roi=is_roi)

#         # # Correción de nube no densa
#         # point_cloud = pcGen.point_cloud_correction(point_cloud, model)
#         # #centroids = pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename)
#         # original_filename = f"{base_filename}_original.ply"
#         # pcGen.save_point_cloud(point_cloud, colors, original_filename)
#         # Generar nube de puntos con filtrado y aplicar DBSCAN
#         point_cloud_list, colors_list, eps, min_samples = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type,  use_roi=is_roi)
#         counter = 0
#         heights = []
#         for pc, cl, letter in zip(point_cloud_list, colors_list, alphabet):
#             # pc = pcGen.point_cloud_correction(pc, model)
#             pc = pcGen.z_correction(pc, model_z)
#             #pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename) #This is DBSCAN process
#             # colors = original_cloud_colors = np.ones_like(point_cloud) * [255, 0, 0]
#             centroids = pcGen.process_point_cloud(pc, eps, min_samples, f"{base_filename}_{letter}")
#             # centroides con profundidad z= 250 (250 es la primera profundidad en donde se empiezan a ver los pies) 
#             # z(centoride) = 250 +- m (m= 30) para un rango aceptable de puntos en donde se puedan tomar puntos sin ruido en la profundidad
#             #  Se define m como un valor multiplicativo inversamente proporcional a la profundidad del centroide
#             #  (e.g: si z(centroide) = 300; m = 25 | si z(centroide) = 350; m = 20 y asi sucesivamente)
#             # TODO: encontrar el valor real de m inical
#             # Posteriormente se necesita evaluar la relacion cambiante de este factor, para esto seria necesario un ajuste lineal
#             # de la siguiente forma m_actual = a*p + b

#             m_initial = 30 #This is an aproximation
#             # optimal_range = [centroids[0][2] - m_initial, centroids[0][2] + m_initial]

#             # AQUI SE NECESITA OBTENER LOS PUNTOS DE point_cloud QUE ESTEN EN EL RAGO OPTIMO APARATIR DE LAS COORDENDAS DEL CENTROIDE
#             if len(centroids)!=0:
#                 # Define el rango óptimo basado en la profundidad del primer centroide
#                 filtered_points = filter_points_by_optimal_range(pc, centroids[0], m_initial)
#                 y_min, y_max = get_Y_bounds(filtered_points)
                
#                 if y_min is not None and y_max is not None:
#                     print(f"Para el centroide con z = {centroids[0][2]}, el rango de Y es: Y_min = {y_min}, Y_max = {y_max}")
#                     # print(f"La altura de la persona {counter+1} es de {abs(y_max - y_min)*4.588771335397176}")
#                     # heights.append(abs(y_max - y_min)*4.588771335397176)
#                     print(f"La altura de la persona {counter+1} es de {abs(y_max - y_min)}")
#                     heights.append(abs(y_max - y_min))
#                     adjusted_situation = get_adjusted_situation(situation, data)

#                     # 4.588771335397176
#                     data.append({
#                         "situation": adjusted_situation + "_" + letter,
#                         "h_estimation": abs(y_max - y_min),
#                         "z_estimation": centroids[0][2]
#                     })
#                 else:
#                     print("No se encontraron puntos en el rango óptimo para este centroide.")
#             counter += 1
#         data_height.append(heights)

#         # z_estimations = [centroid[2] for centroid in centroids] if centroids is not None else []
#         # data.append({
#         #     "situation": situation,
#         #     **{f"z_estimation_{i+1}": z for i, z in enumerate(z_estimations)}
#         # })
#         # heights_estimations = [height for height in heights] if heights is not None else []
#         # data.append({
#         #     "situation": situation + "_" + letter,
#         #     **{f"h_estimation_{i+1}": z for i, z in enumerate(heights_estimations)}
#         # })
      
#     except Exception as e:
#         print(f"Error procesando {situation}: {e}")    

################################################################################################################################

# Flujo principal

# (img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)
# img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
# img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

# disparity = pcGen.compute_disparity(img_l, img_r, configs[camera_type])

# # Generar nube de puntos densa sin filtrado adicional
# dense_point_cloud, dense_colors = pcGen.disparity_to_pointcloud(disparity, Q, img_l)
# dense_point_cloud = dense_point_cloud.astype(np.float64)
# dense_point_cloud = pcGen.point_cloud_correction(dense_point_cloud, model)

# base_filename = f"./point_clouds/prueba/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}_height"

# if not os.path.exists(os.path.dirname(base_filename)):
#     os.makedirs(os.path.dirname(base_filename))
# pcGen.save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

# # Generar nube de puntos con filtrado y aplicar DBSCAN
# point_cloud_list, colors_list, eps, min_samples = pcGen.generate_filtered_point_cloud(img_l, disparity, Q, camera_type,  use_roi=is_roi)
# counter = 0
# for pc, cl in zip(point_cloud_list, colors_list):
#     point_cloud = pcGen.point_cloud_correction(pc, model)
#     #pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename) #This is DBSCAN process
#     # colors = original_cloud_colors = np.ones_like(point_cloud) * [255, 0, 0]
#     centroids = pcGen.process_point_cloud(point_cloud, eps, min_samples, f"{base_filename}_{counter}")
#     # centroides con profundidad z= 250 (250 es la primera profundidad en donde se empiezan a ver los pies) 
#     # z(centoride) = 250 +- m (m= 30) para un rango aceptable de puntos en donde se puedan tomar puntos sin ruido en la profundidad
#     #  Se define m como un valor multiplicativo inversamente proporcional a la profundidad del centroide
#     #  (e.g: si z(centroide) = 300; m = 25 | si z(centroide) = 350; m = 20 y asi sucesivamente)
#     # TODO: encontrar el valor real de m inical
#     # Posteriormente se necesita evaluar la relacion cambiante de este factor, para esto seria necesario un ajuste lineal
#     # de la siguiente forma m_actual = a*p + b

#     m_initial = 30 #This is an aproximation
#     # optimal_range = [centroids[0][2] - m_initial, centroids[0][2] + m_initial]

#     # AQUI SE NECESITA OBTENER LOS PUNTOS DE point_cloud QUE ESTEN EN EL RAGO OPTIMO APARATIR DE LAS COORDENDAS DEL CENTROIDE
#     if len(centroids)!=0:
#         # Define el rango óptimo basado en la profundidad del primer centroide
#         filtered_points = filter_points_by_optimal_range(point_cloud, centroids[0], m_initial)
#         y_min, y_max = get_Y_bounds(filtered_points)
        
#         if y_min is not None and y_max is not None:
#             print(f"Para el centroide con z = {centroids[0][2]}, el rango de Y es: Y_min = {y_min}, Y_max = {y_max}")
#             print(f"La altura de la persona {counter+1} es de {abs(y_max - y_min)*3.3240580662785524}")
#         else:
#             print("No se encontraron puntos en el rango óptimo para este centroide.")
    
    


#     #original_filename = f"{base_filename}_original_{counter}.ply"
#     #pcGen.save_point_cloud(point_cloud, cl, original_filename)
#     counter += 1

# point_cloud, colors = pcGen.roi_no_dense_pc(img_l,disparity,Q)
# print(point_cloud, colors)


################################################################################################################################
pairs = read_image_pairs_by_distance('../images/calibration_results/matlab_1/flexometer')
alphabet = string.ascii_lowercase
alturas = []

def visualize_images(window_name, images, size):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, size[0], size[1])
    cv2.imshow(window_name, images)


for situation, variations in pairs.items():
    try:

        for variation, letter in zip(variations, alphabet):
            print(f"\n \n Procesando situación: {situation} | Variante {letter}")
            #(img_l, img_r), Q = extract_situation_frames(camera_type, situation, False, False)

            MATRIX_Q = configs[camera_type]['MATRIX_Q']
            fs = cv2.FileStorage(MATRIX_Q, cv2.FILE_STORAGE_READ)
            Q = fs.getNode(configs[camera_type]['disparity_to_depth_map']).mat()
            fs.release()

            

            img_l_path = variation[0]
            img_r_path = variation[1]

            # Leer las imágenes con OpenCV y convertirlas a RGB
            img_left = cv2.imread(img_l_path)
            img_right = cv2.imread(img_r_path)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)


            # Calcular la disparidad según el método seleccionado
            if method_used == "SGBM":
                disparity = pcGen.compute_disparity(img_left, img_right, configs[camera_type])
            elif method_used == "RAFT":
                disparity = get_RAFT_disparity_map(
                    restore_ckpt="RAFTStereo/models/raftstereo-middlebury.pth",
                    left_imgs=img_l_path,
                    right_imgs=img_r_path,
                    save_numpy=True
                )
            elif method_used == "SELECTIVE":
                disparity = get_SELECTIVE_disparity_map(
                    restore_ckpt="Selective_IGEV/pretrained_models/middlebury_train.pth",
                    left_imgs=img_l_path,
                    right_imgs=img_r_path,
                    save_numpy=True
                )
            else:
                raise ValueError(f"Método de disparidad no reconocido: {method_used}")

            # Generar nube de puntos densa
            if method_used == "SGBM":
                dense_point_cloud, dense_colors = pcGen.disparity_to_pointcloud(disparity, Q, img_left)
            else:
                dense_point_cloud, dense_colors = pcGen_ML.disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, img_left, use_max_disparity=True)

            # Generar el nombre de archivo base para guardar la nube de puntos
            base_filename = f"./point_clouds/{camera_type}/{mask_type}_disparity/{camera_type}_{situation}_{letter}"

            # Corrección de nube densa (opcional)
            if apply_correction:
                dense_point_cloud = pcGen.point_cloud_correction(dense_point_cloud, model)
                base_filename += "_corregido"

            
            pcGen.save_dense_point_cloud(dense_point_cloud, dense_colors, base_filename)

            # # Generar nube de puntos con filtrado y aplicar DBSCAN
            # point_cloud, colors, eps, min_samples = pcGen.generate_filtered_point_cloud(img_left, disparity, Q, camera_type, use_roi=is_roi)

            # # Correción de nube no densa
            # point_cloud = pcGen.point_cloud_correction(point_cloud, model)
            # #centroids = pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename)
            # original_filename = f"{base_filename}_original.ply"
            # pcGen.save_point_cloud(point_cloud, colors, original_filename)
            # Generar nube de puntos con filtrado y aplicar DBSCAN

            # Generar nube de puntos filtrada
            if method_used == "SGBM":
                point_cloud_list, colors_list, eps, min_samples = pcGen.generate_filtered_point_cloud(
                    img_left, disparity, Q, camera_type, use_roi=is_roi
                )
            else:
                point_cloud_list, colors_list, eps, min_samples = pcGen_ML.generate_filtered_point_cloud(
                    img_left, disparity, fx, fy, cx1, cx2, cy, baseline, camera_type, use_roi=is_roi
                )


            # point_cloud_list.extend(point_cloud_list_RAFT)
            # colors_list.extend(colors_list_RAFT)
            counter = 0
            heights = []
            for pc, cl in zip(point_cloud_list, colors_list):
                # pc = pcGen.point_cloud_correction(pc, model_y, model_z)
                pc = pcGen.z_correction(pc, model)
                #pcGen.process_point_cloud(point_cloud, eps, min_samples, base_filename) #This is DBSCAN process
                # colors = original_cloud_colors = np.ones_like(point_cloud) * [255, 0, 0]
                centroids = pcGen_ML.process_point_cloud(pc, eps, min_samples, f"{base_filename}_person{counter}", cl)
                

                m_initial = 50 #This is an aproximation
                # optimal_range = [centroids[0][2] - m_initial, centroids[0][2] + m_initial]

                # AQUI SE NECESITA OBTENER LOS PUNTOS DE point_cloud QUE ESTEN EN EL RAGO OPTIMO APARATIR DE LAS COORDENDAS DEL CENTROIDE
                if len(centroids)!=0:
                    # Define el rango óptimo basado en la profundidad del primer centroide
                    filtered_points = filter_points_by_optimal_range(pc, centroids[0], m_initial)
                    y_min, y_max = get_Y_bounds(filtered_points)
                    
                    if y_min is not None and y_max is not None:
                        print(f"Para el centroide con z = {centroids[0][2]}, el rango de Y es: Y_min = {y_min}, Y_max = {y_max}")
                        print(f"La altura de la persona {counter+1} es de {abs(y_max - y_min)}")
                        heights.append(abs(y_max - y_min))
                        alturas.append(abs(y_max - y_min))

                        adjusted_situation = get_adjusted_situation(situation+ "_" + letter, data)

                        # adjusted_situation = get_closest_situation(situation, centroids[0][2], data)
                        # 4.588771335397176
                        data.append({
                            "situation": adjusted_situation ,
                            "h_estimation": abs(y_max - y_min),
                            "z_estimation": centroids[0][2]
                        })
                    else:
                        print("No se encontraron puntos en el rango óptimo para este centroide.")
                counter += 1
                
            
                z_estimations = [centroid[2] for centroid in centroids] if centroids is not None else []
                data.append({
                    "situation": situation + "_" + letter,
                    **{f"z_estimation_{i+1}": z for i, z in enumerate(z_estimations)}
                })
            # data_height.append(heights)
            
            
            # heights_estimations = [height for height in heights] if heights is not None else []
            # data.append({
            #     "situation": situation + "_" + letter,
            #     **{f"h_estimation_{i+1}": z for i, z in enumerate(heights_estimations)}
            # })
          
    except Exception as e:
        print(f"Error procesando {situation}: {e}")    

# # alturas = [157.43612581866932, 156.8095422592475, 156.9406790662916, 157.95365574074827, 153.05819018710446, 154.19328141790845, 153.0152661459107, 152.15594113483553, 171.21962589737763, 171.32145118182308, 168.36932374795845, 169.9956647777326, 177.31622114412914, 180.37798739055467, 175.49163042493484, 175.55732353792962, 130.6788474795684, 129.96779483673578, 128.13854669838858, 130.5648414372045, 137.6453896269049, 138.2043374834691, 138.98497561379958, 138.83201464903598, 154.43072825389646, 155.83258743632342, 153.78167300680082, 154.0236285698781, 135.58037823045095, 136.37961246176573, 135.95496231772637, 134.0220086598071, 185.15909915426658, 184.68129879227348, 183.25861312790846, 183.87008285854478, 146.78258888187133, 146.6510012021183, 141.85468447577546, 140.32748268026592, 192.21032622833195, 192.91321279896297, 188.991181423899, 190.16987780147835]
# # graficar_alturas(alturas, 0, 250) 

################################################ GUARDAR DATASET ###############################################################

if len(data) > 0:
    # Guardar dataset como CSV
    # dataset_path = f"../datasets/data/{camera_type}/z_estimation_{camera_type}_{mask_type}_h_validation-LASER2_model-.csv"
    # dataset_path = f"../datasets/data/{camera_type}/validation-z_corrected-LASER2_model-.csv"
    dataset_path = f"../datasets/data/{method_used}_z_estimation_{camera_type}_{mask_type}_train.csv"

    if not os.path.exists(os.path.dirname(dataset_path)):
        os.makedirs(os.path.dirname(dataset_path))

    max_z_count = max(len(row) - 1 for row in data) # -1 porque situation no es una columna z_estimation

    # fieldnames = ["situation"] + [f"z_estimation_{i+1}" for i in range(max_z_count)]
    # fieldnames = ["situation"] + [f"h_estimation_{i+1}" for i in range(max_z_count)]
    fieldnames = ["situation"] + ["z_estimation"] + ["h_estimation"] 


    with open(dataset_path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Dataset guardado en {dataset_path}")




