import os
import cv2
import torch
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import keypoint_extraction as kp

# Aplicar el filtro bilateral
sigma = 1.5  # Parámetro de sigma utilizado para el filtrado WLS.
lmbda = 8000.0  # Parámetro lambda usado en el filtrado WLS.

def save_image(path, image, image_name, grayscale=False):
    if not os.path.exists(path):
        os.makedirs(path)

    files = os.listdir(path)
    image_files = [f for f in files if f.startswith(image_name)]
    next_number = len(image_files) + 1
    new_image_filename = f'{image_name}_{next_number}.png'
    full_path = os.path.join(path, new_image_filename)

    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(full_path, image)

# --------------------------------------------------- DENSE POINT CLOUD ----------------------------------------------------------
def compute_disparity(left_image, right_image, config):
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    blockSize_var = config['blockSize']
    P1 = 8 * 3 * (blockSize_var ** 2)  
    P2 = 32 * 3 * (blockSize_var ** 2) 

    stereo = cv2.StereoSGBM_create(
        numDisparities = config['numDisparities'],
        blockSize = blockSize_var, 
        minDisparity=config['blockSize'],
        P1=P1,
        P2=P2,
        disp12MaxDiff=config['disp12MaxDiff'],
        uniquenessRatio=config['uniquenessRatio'],
        preFilterCap=config['preFilterCap'],
        mode=config['mode']
    )

    left_disp = stereo.compute(left_image, right_image)
    right_matcher = cv2.ximgproc.createRightMatcher(stereo)
    right_disp = right_matcher.compute(right_image, left_image)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    filtered_disp = wls_filter.filter(left_disp, left_image, disparity_map_right=right_disp)
    return filtered_disp

def disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, image, custom_mask=None):
    depth = (fx * baseline) / (-disparity + (cx2 - cx1))
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))

    points_grid = np.stack(((xx - cx1) / fx, (yy - cy) / fy, np.ones_like(xx)), axis=-1) * depth[:, :, np.newaxis]

    mask = np.ones((H, W), dtype=bool)

    # AQUI TE QUEDASTE COJUDO
    mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = True
    mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = True

    if custom_mask is not None:
        mask = custom_mask < 0

    out_points = points_grid[mask].astype(np.float64)
    out_colors = image[mask].astype(np.float64)

    return out_points, out_colors

def apply_dbscan(point_cloud, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(point_cloud)
    labels = db.labels_
    return labels

def get_centroids(point_cloud, labels):
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    if not unique_labels:
        print("No hay clusters.")
        return None
    else:
        centroids = []
        for label in unique_labels:
            cluster_points = point_cloud[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
            print("z = ", str(centroid[2]))
        return np.array(centroids)

def create_point_cloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    return pcd

def save_point_cloud(point_cloud, colors, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pcd = create_point_cloud(point_cloud, colors)
    o3d.io.write_point_cloud(filename, pcd, print_progress=True)

def process_point_cloud(point_cloud, eps, min_samples, base_filename):
    labels = apply_dbscan(point_cloud, eps, min_samples)
    centroids = get_centroids(point_cloud, labels)

    if centroids is not None:
        centroid_colors = np.tile([[255, 0, 0]], (len(centroids), 1))
        centroid_filename = f"{base_filename}_centroids.ply"
        save_point_cloud(centroids, centroid_colors, centroid_filename)

    original_cloud_colors = np.ones_like(point_cloud) * [0, 0, 255]
    original_filename = f"{base_filename}_original.ply"
    save_point_cloud(point_cloud, original_cloud_colors, original_filename)

    return centroids

def generate_all_filtered_point_cloud(img_l, disparity, fx, fy, cx1, cx2, cy, baseline, camera_type, use_roi=True):
    if use_roi:
        seg = kp.get_segmentation(img_l)
        result_image = kp.apply_seg_mask(disparity, seg)
        eps, min_samples = 2, 3500
    else:
        keypoints = kp.get_keypoints(img_l)
        result_image = kp.apply_keypoints_mask(disparity, keypoints)
        eps = 50 if "matlab" in camera_type else 10
        min_samples = 6

    point_cloud, colors = disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, img_l, result_image)
    return point_cloud, colors, eps, min_samples

def generate_filtered_point_cloud(img_l, disparity, fx, fy, cx1, cx2, cy, baseline, camera_type, use_roi=True):
    result_image_list = []
    point_cloud_list = []
    colors_list = []

    if use_roi:
        seg = kp.get_segmentation(img_l)
        for i in seg:
            i_list = [i]
            result_image = kp.apply_seg_mask(disparity, i_list)
            result_image_list.append(result_image)
        eps, min_samples = 5, 1000
    else:
        keypoints = kp.get_keypoints(img_l)
        for i in keypoints:
            i_list = [i]
            result_image = kp.apply_keypoints_mask(disparity, i_list)
            result_image_list.append(result_image)
        eps = 300 if "matlab" in camera_type else 10
        min_samples = 6

    for mask in result_image_list:
        point_cloud, colors = disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, img_l, mask)
        point_cloud_list.append(point_cloud)
        colors_list.append(colors)
    
    return point_cloud_list, colors_list, eps, min_samples

def roi_no_dense_pc(img_l, disparity, fx, fy, cx1, cx2, cy, baseline):
    segmentation = kp.get_segmentation(img_l)
    point_clouds, pc_colors = [], []

    for seg in segmentation:
        mask = kp.apply_seg_individual_mask(img_l, seg)
        point_cloud, color = disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, img_l, mask)
        point_clouds.append(point_cloud)
        pc_colors.append(color)

    return point_clouds, pc_colors

def roi_source_point_cloud(img_l, img_r, fx, fy, cx1, cx2, cy, baseline, config):
    eps, min_samples = 5, 1800

    roi_left = kp.get_roi(img_l)
    roi_right = kp.get_roi(img_r)

    result_img_left = kp.apply_roi_mask(img_l, roi_left)
    result_img_right = kp.apply_roi_mask(img_r, roi_right)

    disparity = compute_disparity(result_img_left, result_img_right, config)
    filtered_disparity = kp.apply_roi_mask(disparity, roi_left)

    dense_point_cloud, dense_colors = disparity_to_pointcloud(disparity, fx, fy, cx1, cx2, cy, baseline, img_l, filtered_disparity)

    return filtered_disparity, dense_point_cloud, dense_colors, eps, min_samples

def point_cloud_correction(points, model):
    points = np.asarray(points)

    x = points[:, 0].reshape(-1,1)
    x_pred = model.predict(x)
    y = points[:, 1].reshape(-1,1)
    y_pred = model.predict(y)
    z = points[:, 2].reshape(-1,1)
    z_pred = model.predict(z)

    corrected_points = np.column_stack((x_pred, y_pred, z_pred))
    return corrected_points

def save_dense_point_cloud(point_cloud, colors, base_filename):
    if not os.path.exists(os.path.dirname(base_filename)):
        os.makedirs(os.path.dirname(base_filename))
    dense_filename = f"{base_filename}_dense.ply"
    save_point_cloud(point_cloud, colors, dense_filename)
