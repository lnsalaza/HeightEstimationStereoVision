# HeightEstimationStereoVision

This project focuses on stereo vision to generate dense depth maps and segment people in images and videos. Using OpenCV, the project rectifies the input images or videos and generates a depth map. The result is a dense cloud of scenes. Various machine learning models, including RAFT-Stereo and Selective-IGEV, are now used to enhance disparity map generation.

## Features

- **Image and Video Rectification**: Rectify input images and videos for accurate stereo vision processing.
- **Depth Map Generation**: Generate dense depth maps using OpenCV, RAFT-Stereo, and Selective-IGEV.
- **Person Segmentation**: Use YOLOv8 for detecting and segmenting people in the scene.
- **3D Projection**: Perform 2D to 3D projection using YOLO results as a mask.
- **Dense to Sparse Cloud Conversion**: Convert dense clouds to sparse clouds focusing on key points and ROIs.
- **Depth and Height Measurement**: Calculate depth and height of detected people using their key points or entire ROI.
- **API Integration**: Expose functionalities via a FastAPI for easier access and testing.
- **Orchestrator System**: A dynamic system that allows switching between tasks such as dense cloud generation, height estimation, or feature extraction in real-time.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/lnsalaza/HeightEstimationStereoVision.git
    cd HeightEstimationStereoVision
    ```

2. Install the required packages:
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

## Usage

### Running the API

You can run the FastAPI server to expose the various functionalities of this project.

1. To start the API server:
    ```bash
    uvicorn api:app --reload
    ```

2. Once started, you can interact with the API by navigating to the following URL:
    ```
    http://127.0.0.1:8000/docs
    ```
    This will bring up the Swagger documentation where you can test the different API endpoints.

3. Some example API endpoints include:
   - `/generate_point_cloud/dense/`: Generate a dense 3D point cloud from stereo images.
   - `/generate_point_cloud/nodense/height_estimation/`: Estimate the height of a person detected in the point cloud.
   - `/get_profiles/`: Retrieve a list of all calibration profiles available.

### Running the Demo with Orchestrator and YOLOv8

You can use the **orchestrator** to dynamically switch between tasks such as height estimation, dense cloud generation, or feature extraction while using live images from a webcam.

1. To run the demo:
    ```bash
    python demo.py
    ```

2. This demo utilizes YOLOv8 to detect persons via webcam and sends the detected images to the orchestrator for height estimation or other processes, depending on the requirements. You can switch the task in real-time.

### Using the Orchestrator

The orchestrator is a system that allows switching between different operations dynamically, such as:

- **Dense point cloud generation**.
- **Sparse point cloud generation**.
- **Feature extraction**.
- **Height estimation**.

#### Example: Using the Orchestrator in Real-Time
```python
from orchestrator.orchestrator import Orchestrator

# Initialize with images and the desired operation (e.g., 'height')
orchestrator = Orchestrator(img_left=left_image_array, img_right=right_image_array, requirement="height")

# Set new images dynamically
orchestrator.set_images(new_left_image, new_right_image)

# Execute the task (e.g., estimating height)
result = orchestrator.execute()
print(f"Result: {result}")
```

### YOLOv8 Detection with Orchestrator

The `demo.py` script integrates YOLOv8 for person detection with the orchestrator. After detecting a person, the system can estimate the person's height using the stereo vision setup.

```python
python demo.py
```

The demo will:

- Detect people using YOLOv8.
- Estimate their height by processing the stereo images using the orchestrator.
- Display the result in the terminal while continuing the video feed.

### Calibration

1. **Calibrate Images**:
    ```bash
    python calibration.py --json ../config_files/matlab_1/stereoParameters.json --xml ../config_files/matlab_1/newStereoMap.xml --img_left ../images/originals/IMG_LEFT.jpg --img_right ../images/originals/IMG_RIGHT.jpg
    ```

2. **Rectify Images/Videos**:
    ```bash
    python rectification.py --input_type image --input_images_folder ../images/laser/groundTruth --output_images_folder ../images/calibration_results/matlab_1 --xml ../config_files/matlab_1/newStereoMap.xml
    ```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project does not currently have an official license.

## Contact

For any questions or suggestions, feel free to open an issue or contact my colleague or me directly.
