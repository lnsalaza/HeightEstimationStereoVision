import numpy as np 
import cv2
 
# left_video = 'left.avi'
# right_video = 'right.avi'

left_video = '../videos/rectified/left_rectified_matlab_2.avi'
right_video = '../videos/rectified/right_rectified_matlab_2.avi'

# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 2 # Camera ID for left camera
CamR_id = 0 # Camera ID for right camera
 # Aplicar el filtro bilateral
sigma = 1.8  # Parámetro de sigma utilizado para el filtrado WLS.
lmbda = 7500.0  # Parámetro lambda usado en el filtrado WLS.




CamL= cv2.VideoCapture(left_video)
CamR = cv2.VideoCapture(right_video)

# Reading the mapping values for stereo image rectification
# cv_file = cv2.FileStorage("stereoMap.xml", cv2.FILE_STORAGE_READ)
# Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
# Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
# Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
# Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
# cv_file.release()
 
def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',6,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing) 
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,63,nothing) #5, 62
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,50,nothing) #25
 


# FRAME EXTRACT

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        retval, frame = cap.read()
        if not retval:
            break
    
        frames.append(frame)
    
    cap.release()
    return frames



#TODO: HSI


# Creating an object of StereoBM algorithm


while True:
 
  # Capturing and storing left and right camera images
  stereo = cv2.StereoSGBM_create(
     numDisparities = 68,
        blockSize = 12, 
        minDisparity=12,

        disp12MaxDiff=0,
        uniquenessRatio=10,
        speckleWindowSize=0,
        preFilterCap=6,
        mode= cv2.StereoSGBM_MODE_HH
  )
  retL, imgL= CamL.read()
  retR, imgR= CamR.read()

   
  # Proceed only if the frames have been captured
  if retL and retR:
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
 
    # # Applying stereo image rectification on the left image
    # Left_nice= cv2.remap(imgL_gray,
    #           Left_Stereo_Map_x,
    #           Left_Stereo_Map_y,
    #           cv2.INTER_LANCZOS4,
    #           cv2.BORDER_CONSTANT,
    #           0)
     
    # # Applying stereo image rectification on the right image
    # Right_nice= cv2.remap(imgR_gray,
    #           Right_Stereo_Map_x,
    #           Right_Stereo_Map_y,
    #           cv2.INTER_LANCZOS4,
    #           cv2.BORDER_CONSTANT,
    #           0)
 
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16 # 16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5 # 53
    #preFilterType = cv2.getTrackbarPos('preFilterType','disp') # 1
    #preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5 # 9 
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp') # 63
    #textureThreshold = cv2.getTrackbarPos('textureThreshold','disp') # 10
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp') # 15
    speckleRange = cv2.getTrackbarPos('speckleRange','disp') # 0
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2 # 6
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp') # 5
    minDisparity = cv2.getTrackbarPos('minDisparity','disp') # 14
     
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    #stereo.setPreFilterType(preFilterType)
    #stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    #stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calcular el mapa de disparidad de la imagen izquierda a la derecha
    left_disp = stereo.compute(imgL_gray, imgR_gray).astype(np.float32) / 16.0

    # Crear el matcher derecho basado en el matcher izquierdo para consistencia
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)

    # Calcular el mapa de disparidad de la imagen derecha a la izquierda
    right_disp = right_matcher.compute(imgR_gray, imgL_gray).astype(np.float32) / 16.0

    # Crear el filtro WLS y configurarlo
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    # Filtrar el mapa de disparidad utilizando el filtro WLS
    filtered_disp = wls_filter.filter(left_disp, imgL_gray, disparity_map_right=right_disp)

    # Normalización para la visualización o procesamiento posterior
    filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filtered_disp = np.uint8(filtered_disp)

    # Displaying the disparity map
    cv2.imshow("disp",filtered_disp)
 
    # Close window using esc key
    if cv2.waitKey(75) == 27:
      break
   
  else:
    CamL= cv2.VideoCapture(CamL_id)
    CamR= cv2.VideoCapture(CamR_id)