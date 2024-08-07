Aquí tienes el README actualizado, incluyendo las funciones de prueba como demos para mostrar el uso del sistema:

---

# HeightEstimationStereoVision

This project focuses on stereo vision to generate dense depth maps and segment people in images and videos. Using OpenCV, the project rectifies the input images or videos and generates a depth map. The result is a dense cloud of scenes. Various machine learning models, including RAFT-Stereo and Selective-IGEV, are now used to enhance disparity map generation.

## Features

- **Image and Video Rectification**: Rectify input images and videos for accurate stereo vision processing.
- **Depth Map Generation**: Generate dense depth maps using OpenCV, RAFT-Stereo, and Selective-IGEV.
- **Person Segmentation**: Use YOLOv8 for detecting and segmenting people in the scene.
- **3D Projection**: Perform 2D to 3D projection using YOLO results as a mask.
- **Dense to Sparse Cloud Conversion**: Convert dense clouds to sparse clouds focusing on key points and ROIs.
- **Depth and Height Measurement**: Calculate depth and height of detected people using their key points or entire ROI.

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/lnsalaza/HeightEstimationStereoVision.git
    cd HeightEstimationStereoVision
    ```

2. Install the required packages:
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

## Usage

### Calibration

1. **Calibrate Images**:
    ```
    python calibration.py --json ../config_files/matlab_1/stereoParameters.json --xml ../config_files/matlab_1/newStereoMap.xml --img_left ../images/originals/IMG_LEFT.jpg --img_right ../images/originals/IMG_RIGHT.jpg
    ```

2. **Rectify Images/Videos**:
    ```
    python rectification.py --input_type image --input_images_folder ../images/laser/groundTruth --output_images_folder ../images/calibration_results/matlab_1 --xml ../config_files/matlab_1/newStereoMap.xml
    ```

### Depth Map and Point Cloud Generation

The following examples demonstrate how to use the provided functions to generate disparity maps and point clouds. These examples are included in the script `pc_densa.py`:

#### Example: Test Disparity Map
```python
def test_disparity_map(img_left, img_right, config, method):
    disparity_map = compute_disparity(img_left, img_right, config, method)
    plt.imshow(disparity_map, cmap='jet')
    plt.colorbar()
    plt.title('Disparity Map')
    plt.show()

# Usage
test_disparity_map(img_left, img_right, config, method='SELECTIVE')
```

#### Example: Test Dense Point Cloud
```python
def test_point_cloud(img_left, img_right, config, method, use_max_disparity):
    point_cloud, colors = generate_dense_point_cloud(img_left, img_right, config, method, use_max_disparity)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/densaDEMO")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="3D Point Cloud", width=800, height=600)
    viewer.add_geometry(pcd)
    viewer.get_render_option().point_size = 1
    viewer.run()
    viewer.destroy_window()

# Usage
test_point_cloud(img_left, img_right, config, method='RAFT', use_max_disparity=False)
```

#### Example: Test Filtered Point Cloud
```python
def test_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity):
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud", width=800, height=600)
    viewer.add_geometry(pcd)
    viewer.get_render_option().point_size = 5 if not use_roi else 1
    viewer.run()
    viewer.destroy_window()

# Usage
test_filtered_point_cloud(img_left, img_right, config, method='SELECTIVE', use_roi=False, use_max_disparity=True)
```

#### Example: Test Filtered Point Cloud with Centroids
```python
def test_filtered_point_cloud_with_centroids(img_left, img_right, config, method, use_roi, use_max_disparity):
    point_cloud, colors = generate_combined_filtered_point_cloud(img_left, img_right, config, method, use_roi, use_max_disparity)
    pcGen.save_point_cloud(point_cloud, colors, "./point_clouds/DEMO/NOdensaDEMO")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    centroids = compute_centroids(point_cloud)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(window_name="Filtered 3D Point Cloud with Centroids", width=800, height=600)
    viewer.add_geometry(pcd)
    for centroid in centroids:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        viewer.add_geometry(sphere)
    viewer.get_render_option().point_size = 5 if not use_roi else 1
    viewer.run()
    viewer.destroy_window()

# Usage
test_filtered_point_cloud_with_centroids(img_left, img_right, config, method='RAFT', use_roi=False, use_max_disparity=True)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project does not currently have an official license.

## Contact

For any questions or suggestions, feel free to open an issue or contact my colleague or me directly.

