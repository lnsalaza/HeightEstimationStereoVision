import cv2
import os
import numpy as np
import random
import plotly.graph_objs as go

master_path_to_dataset = "../images/calibration_results/matlab_1/flexometer/150/" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

# fixed camera parameters for this stereo setup (from calibration)

# camera_focal_length_px = (1429.4995220185822 + 1433.6695087748499) / 2  # focal length in pixels
camera_focal_length_px = 1429.4995220185822
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.58     # camera baseline in metres

image_centre_h = 600.8227256572083 
image_centre_w = (506.4722541384677 + 520.1168815891416)/2

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):
    points = []
    f = camera_focal_length_px
    B = stereo_camera_baseline_m
    height, width = disparity.shape[:2]

    for y in range(height): 
        for x in range(width): 
            if (disparity[y, x] > 0):
                Z = (f * B) / disparity[y, x]
                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                if(rgb.size > 0):
                    points.append([X, Y, Z, rgb[y, x, 2], rgb[y, x, 1], rgb[y, x, 0]])
                else:
                    points.append([X, Y, Z])
    return points

def visualizar_nube_de_puntos(points):
    
    points = np.array(points)
    xyz_points = points[:, :3]
    colors = points[:, 3:6] / 255.0  # Normalizar los colores al rango 0-1

    origin_point = np.array([[0, 0, 0]])
    origin_color = np.array([[1, 0, 0]])  # Color rojo

    points = np.vstack([xyz_points, origin_point])
    colors = np.vstack([colors, origin_color])

    NUM_POINTS_TO_DRAW = 640000
    subset = np.random.choice(points.shape[0], size=(NUM_POINTS_TO_DRAW - 1,), replace=False)
    subset = np.append(subset, points.shape[0] - 1)  # Asegurar que el origen esté incluido
    points_subset = points[subset]
    colors_subset = colors[subset]

    x, y, z = points_subset.T

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
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

# resolve full directory location of data set for left / right images

full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

full_path_filename_left = os.path.join(full_path_directory_left, "1506942480.483420_L.png")
full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R")

# max_disparity = 128


blockSize = 7
min_disparity = 5
max_disparity = 80
P1 = 8
P2 = 32

stereoProcessor = cv2.StereoSGBM_create(
	# minDisparity=min_disparity,
	numDisparities=(max_disparity - min_disparity),
	blockSize=blockSize,
	P1=3*blockSize*blockSize * P1,
	P2=3*blockSize*blockSize * P2,
	disp12MaxDiff=33,
	preFilterCap=33,
	uniquenessRatio=10,
#	speckleWindowSize=100,
	speckleRange=1,
#	mode=cv.StereoSGBM_MODE_SGBM
# 	mode=cv.StereoSGBM_MODE_HH
	mode=cv2.StereoSGBM_MODE_HH
)



stereoProcessorR = cv2.ximgproc.createRightMatcher(stereoProcessor)

# WLS filter setup
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereoProcessor)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

print(full_path_filename_left)
print(full_path_filename_right)

if True:
    imgL = cv2.imread("../images/calibration_results/matlab_1/flexometer/150/14_03_37_13_05_2024_IMG_LEFT.jpg", cv2.IMREAD_COLOR)
    imgR = cv2.imread("../images/calibration_results/matlab_1/flexometer/150/14_03_37_13_05_2024_IMG_RIGHT.jpg", cv2.IMREAD_COLOR)

    print("-- files loaded successfully")

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    disparityL = stereoProcessor.compute(grayL, grayR)
    disparityR = stereoProcessorR.compute(grayR, grayL)

    # Apply WLS filter
    filteredDisparity = wls_filter.filter(disparityL, grayL, disparity_map_right=disparityR)

    dispNoiseFilter = 5 
    cv2.filterSpeckles(filteredDisparity, 0, 4000, max_disparity - dispNoiseFilter)

    _, filteredDisparity = cv2.threshold(filteredDisparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (filteredDisparity/16.)

    #cv2.imshow("disparity", (disparity_scaled * (255. / max_disparity)).astype(np.uint8))

    points = project_disparity_to_3d(disparity_scaled, max_disparity, imgL)

    visualizar_nube_de_puntos(points)

else:
    print("-- files skipped (perhaps one is missing or path is wrong)")
    print()

cv2.destroyAllWindows()
