import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# left_image = cv.imread('img_l.png', cv.IMREAD_GRAYSCALE)
# right_image = cv.imread('img_r.png', cv.IMREAD_GRAYSCALE)


# # left_image = cv.imread('tsukuba_l.png', cv.IMREAD_GRAYSCALE)
# # right_image = cv.imread('tsukuba_r.png', cv.IMREAD_GRAYSCALE)

# left_image_cesar = cv.imread('img_l-transformed.png', cv.IMREAD_GRAYSCALE)
# right_image_cesar = cv.imread('img_r-transformed.png', cv.IMREAD_GRAYSCALE)

# # También corregí la lectura de imágenes adicionales
# left_image = cv.fastNlMeansDenoising(left_image, None, 10, 7, 21)
# right_image = cv.fastNlMeansDenoising(right_image, None, 10, 7, 21)



# stereo = cv.StereoBM_create(numDisparities = 16*9, blockSize=25)

# depth = stereo.compute(left_image, right_image)


# cv.imwrite(depth)

# cv.imshow("Left", left_image)
# cv.imshow("Right", right_image)

# plt.imshow(depth)
# plt.axis("off")
# plt.show()




import os

# Asegúrate de que las imágenes están cargadas en escala de grises y rectificadas
left_image = cv.imread('image_r_color.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('image_l_color.png', cv.IMREAD_GRAYSCALE)

def rotate_image(image, angle):
    

    # Construct the rotation matrix
    center = (image.shape[1] // 2, image.shape[0] // 2)

    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale=1)

    # Apply the rotation
    image = cv.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return image


rotated_image_l = rotate_image(left_image, 180)
rotated_image_r = rotate_image(right_image, 180)
           

# Save the rotated image
cv.imwrite("rotated_image_l.jpg", rotate_image(left_image, 180))
cv.imwrite("rotated_image_r.jpg", rotate_image(right_image, 180))


# # Non-Local Means
# left_image = cv.fastNlMeansDenoising(left_image, None, 10, 7, 21)
# right_image = cv.fastNlMeansDenoising(right_image, None, 10, 7, 21) 



# FILTRO BILATERAL
# left_image = cv.bilateralFilter(left_image, 9, 75, 75)
# right_image = cv.bilateralFilter(right_image, 9, 75, 75)


# # BLUR
# left_image = cv.GaussianBlur(left_image, (11, 11), 0)
# right_image = cv.GaussianBlur(right_image, (11, 11), 0)

# Directorio para guardar los resultados
output_directory = 'depth_maps'




if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Configuraciones para probar
numDisparities_values = [16 * i for i in range(1, 15, 2)]  # Valores de numDisparities como múltiplos de 16
blockSize_values = [2*i+5 for i in range(1, 30,2)]  # Diferentes valores de blockSize
depth = None 
# Ciclo sobre cada combinación de numDisparities y blockSize
for numDisparities in numDisparities_values:
    for blockSize in blockSize_values:
        if blockSize % 2 == 1:  # blockSize debe ser impar
            # Crear el objeto StereoBM o a su vez StereoSGBM para cada configuración
            stereo = cv.StereoBM_create(
             
                numDisparities=numDisparities,
                blockSize=blockSize,
                
            )
        
            stereo.setPreFilterType(1)
            stereo.setPreFilterSize(15)
            stereo.setPreFilterCap(63)
            stereo.setTextureThreshold(10)
            stereo.setUniquenessRatio(15)
            stereo.setSpeckleRange(0)
            stereo.setSpeckleWindowSize(6)
            stereo.setDisp12MaxDiff(5)
            stereo.setMinDisparity(10)

            # Calcular el mapa de profundidad
            depth = stereo.compute(rotated_image_l, rotated_image_r )
            #depth = stereo.compute(left_image, right_image)
        
            

            norm_coef = 255/depth.max() 
            
            # Construir nombre de archivo
            filename = f'depth_numDisp_{numDisparities}_blockSize_{blockSize}.png'

            # Guardar el mapa de profundidad en el directorio especificado
            cv.imwrite(os.path.join(output_directory, filename),depth )

            # print(f'Mapa de profundidad guardado: {filename}')
