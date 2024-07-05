import os
import cv2
import csv
import json
import torch
import numpy as np
import open3d as o3d
import cv2

img = cv2.imread("../images/calibration_results/image_l.png")

torch.cuda.set_device(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the parameters into the model
model = torch.load("mocha-V2.pth")



print(model)