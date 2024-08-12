import os
import cv2
import argparse
import numpy as np
import argparse
import sys

INPUT_TYPE = "image"
LEFT_INPUT_VIDEO_PATH = "../videos/original/original_left.avi"
RIGHT_INPUT_VIDEO_PATH = "../videos/original/original_right.avi"
LEFT_OUTPUT_VIDEO_PATH = "../videos/rectified/left_rectified.avi"
RIGHT_OUTPUT_VIDEO_PATH = "../videos/rectified/right_rectified.avi"
DEFAULT_JSON = "../config_files/matlab_1/stereoParameters.json"
DEFAULT_XML = "../config_files/matlab_1/newStereoMap.xml"

GROUND_TRUTH_INPUT_IMAGES_PATH = '../images/originals/h_validation'
GROUND_TRUTH_OUTPUT_IMAGES_PATH = '../images/calibration_results/matlab_1/h_validation'


def get_stereo_map_parameter(file_name, parameter):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    ret = fs.getNode(parameter).mat()
    fs.release()
    return ret

def get_stereo_map_parameters(file_name):
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    stereoMapL_x = fs.getNode("stereoMapL_x").mat()
    stereoMapL_y = fs.getNode("stereoMapL_y").mat()
    stereoMapR_x = fs.getNode("stereoMapR_x").mat()
    stereoMapR_y = fs.getNode("stereoMapR_y").mat()

    fs.release()
    return stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y

def calibrate_video(left_video_path, right_video_path, left_output_video, right_output_video, xml_file):
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = get_stereo_map_parameters(xml_file)

    cap_left = cv2.VideoCapture(left_video_path)
    cap_right = cv2.VideoCapture(right_video_path)

    total_frames_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = min(total_frames_left, total_frames_right)

    width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_left.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_left = cv2.VideoWriter(left_output_video, fourcc, fps, (width, height))
    out_right = cv2.VideoWriter(right_output_video, fourcc, fps, (width, height))

    i = 0
    while cap_left.isOpened() and cap_right.isOpened():
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_right.read()

        if not (ret_l and ret_r):
            break

        rectified_frame_l = cv2.remap(frame_l, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
        rectified_frame_r = cv2.remap(frame_r, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

        out_left.write(rectified_frame_l)
        out_right.write(rectified_frame_r)

        # Update progress
        progress = int((i / total_frames) * 20)  # 20 segments in the progress bar
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%" % ('='*progress, 5*progress))
        sys.stdout.flush()

        i += 1

    # Ensure the final 100% state is reached and displayed
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] 100%%" % ('='*20))
    sys.stdout.flush()
    print()  # Move to the next line after progress bar completes

    cap_left.release()
    cap_right.release()
    out_left.release()
    out_right.release()

def remap_image(image_path, map1, map2):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    rectified_image = cv2.remap(image, map1, map2, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    return rectified_image

def calibrate_images(input_path, output_path, xml_file):
    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = get_stereo_map_parameters(xml_file)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for root, _, files in os.walk(input_path):
        for file in files:
            if 'LEFT' in file.upper():
                image_path = os.path.join(root, file)
                rectified_image = remap_image(image_path, stereoMapL_x, stereoMapL_y)
                save_path = os.path.join(output_path, os.path.relpath(root, input_path), file)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, rectified_image)
            elif 'RIGHT' in file.upper():
                image_path = os.path.join(root, file)
                rectified_image = remap_image(image_path, stereoMapR_x, stereoMapR_y)
                save_path = os.path.join(output_path, os.path.relpath(root, input_path), file)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, rectified_image)

            
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate video frames using stereo camera maps.")
    parser.add_argument("--input_type", type=str, default=INPUT_TYPE, help="\"video\", if input is a video. \"image\", if inputs are images. Default: \"video\".")
    parser.add_argument("--input_left", type=str, default=LEFT_INPUT_VIDEO_PATH, help="Left input video file path.")
    parser.add_argument("--input_right", type=str, default=RIGHT_INPUT_VIDEO_PATH, help="Right input video file path.")
    parser.add_argument("--output_left", type=str, default=LEFT_OUTPUT_VIDEO_PATH, help="Left output video file path.")
    parser.add_argument("--output_right", type=str, default=RIGHT_OUTPUT_VIDEO_PATH, help="Right output video file path.")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML, help="XML file with stereo map parameters.")
    parser.add_argument("--input_images_folder", type=str, default=GROUND_TRUTH_INPUT_IMAGES_PATH, help="Path of the folder that contains input images.")
    parser.add_argument("--output_images_folder", type=str, default=GROUND_TRUTH_OUTPUT_IMAGES_PATH, help="Path of the folder that will contain calibrated images.")
    args = parser.parse_args()
    
    # Call the calibration function with command line arguments
    if(args.input_type == 'image'):
        calibrate_images(args.input_images_folder, args.output_images_folder, args.xml)
    else: 
        calibrate_video(args.input_left, args.input_right, args.output_left, args.output_right, args.xml)