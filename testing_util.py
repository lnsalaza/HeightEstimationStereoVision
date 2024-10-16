import os
import cv2
import json
import api_util.image_utils as imgutil
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


def visualize_images(images, window_name='VISUALIZE IMAGE' , size=(1920,1080)):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, size[0], size[1])
    cv2.imshow(window_name, images)


def load_config(path):
    """
    Carga la configuración desde un archivo JSON.
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def convert_to_gray(input_path, output_path):
    # Load the image from the input path
    image = cv2.imread(input_path)

    # Check if the image was loaded correctly
    if image is None:
        raise ValueError(f"Error loading the image from {input_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image to the output path
    cv2.imwrite(output_path, gray_image)

    print(f"Image converted to grayscale and saved at {output_path}")

def test_convert_video():
    input_file = '15_10_2024_10_28_38_LEFT.webm'
    output_file = '15_10_2024_10_28_38_LEFT.avi'
    imgutil.convert_video(input_file, output_file)

    print(f"Video converted saved at {output_file}")