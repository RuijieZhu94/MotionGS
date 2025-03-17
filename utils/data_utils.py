import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset:
    def __init__(self, viewpoint_stack, white_background):
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        self.resolution = viewpoint_stack[0].resolution
    
    def load_image(self, image_path):
        with Image.open(image_path) as image_load:
            im_data = np.array(image_load.convert("RGBA"))
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
        image_load = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        resized_image_rgb = PILtoTorch(image_load, self.resolution)
        viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
        if resized_image_rgb.shape[0] == 4:
            gt_alpha_mask = resized_image_rgb[3:4, ...]
            viewpoint_image *= gt_alpha_mask
        else:
            viewpoint_image *= torch.ones((1, self.resolution[1], self.resolution[0]))

        return viewpoint_image
