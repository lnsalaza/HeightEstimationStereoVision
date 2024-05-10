import cv2
import argparse
import numpy as np


LEFT_INPUT_VIDEO_PATH = "../videos/original/original_left.avi"
RIGHT_INPUT_VIDEO_PATH = "../videos/original/original_right.avi"
LEFT_OUTPUT_VIDEO_PATH = "../videos/rectified/left_rectified.avi"
RIGHT_OUTPUT_VIDEO_PATH = "../videos/rectified/right_rectified.avi"
DEFAULT_JSON = "../config_files/stereoParameters.json"
DEFAULT_XML = "../config_files/newStereoMap.xml"


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

def calibrate_video(left_video_path, right_video_path , left_output_video, right_output_video, xml_file):

    stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y = get_stereo_map_parameters(xml_file)

    cap_left = cv2.VideoCapture(left_video_path)
    cap_righ = cv2.VideoCapture(right_video_path)

    width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap_left.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out_left = cv2.VideoWriter(left_output_video, fourcc, fps, (width, height))
    out_right = cv2.VideoWriter(right_output_video, fourcc, fps, (width, height))

    i = 0
    while cap_left.isOpened() & cap_righ.isOpened():
        print(i)
        ret_l, frame_l = cap_left.read()
        ret_r, frame_r = cap_righ.read()

        if not (ret_l or ret_r):
            break

        rectified_frame_l = cv2.remap(frame_l, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        rectified_frame_r = cv2.remap(frame_r, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        out_left.write(rectified_frame_l)
        out_right.write(rectified_frame_r)

        i=i+1

    cap_left.release()
    cap_righ.release()

    out_left.release()
    out_right.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate video frames using stereo camera maps.")
    parser.add_argument("--input_left", type=str, default=LEFT_INPUT_VIDEO_PATH, help="Left input video file path.")
    parser.add_argument("--input_right", type=str, default=RIGHT_INPUT_VIDEO_PATH, help="Right input video file path.")
    parser.add_argument("--output_left", type=str, default=LEFT_OUTPUT_VIDEO_PATH, help="Left output video file path.")
    parser.add_argument("--output_right", type=str, default=RIGHT_OUTPUT_VIDEO_PATH, help="Right output video file path.")
    parser.add_argument("--xml", type=str, default=DEFAULT_XML, help="XML file with stereo map parameters.")
    
    args = parser.parse_args()
    
    # Call the calibration function with command line arguments
    calibrate_video(args.input_left, args.input_right, args.output_left, args.output_right, args.xml)