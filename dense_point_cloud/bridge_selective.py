import sys
from pathlib import Path

# Añade el directorio donde se encuentra el módulo 'core' al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent / 'Selective_IGEV'))

from Selective_IGEV.demo_imgs import demo as Selective_demo


def get_SELECTIVE_disparity_map(restore_ckpt="Selective_IGEV/pretrained_models/middlebury_train.pth", 
                                left_imgs="Selective_IGEV/datasets/CIDIS/Laser_ground_truth/250_350/15_18_56_07_06_2024_IMG_LEFT.jpg", 
                                right_imgs="Selective_IGEV/datasets/CIDIS/Laser_ground_truth/250_350/15_18_56_07_06_2024_IMG_RIGHT.jpg", 
                                output_directory="seletive_demo_output", save_numpy=False, mixed_precision=False, valid_iters=32, 
                                hidden_dims=[128]*3, corr_implementation="reg", shared_backbone=False, corr_levels=2, 
                                corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3, max_disp=192):
    
    class Args:
        def __init__(self):
            self.restore_ckpt = restore_ckpt
            self.save_numpy = save_numpy
            self.left_imgs = left_imgs
            self.right_imgs = right_imgs
            self.output_directory = output_directory
            self.mixed_precision = mixed_precision
            self.valid_iters = valid_iters
            self.hidden_dims = hidden_dims
            self.corr_implementation = corr_implementation
            self.shared_backbone = shared_backbone
            self.corr_levels = corr_levels
            self.corr_radius = corr_radius
            self.n_downsample = n_downsample
            self.slow_fast_gru = slow_fast_gru
            self.n_gru_layers = n_gru_layers
            self.max_disp = max_disp
    
    args = Args()

    disparities = Selective_demo(args)
    return disparities[0]

