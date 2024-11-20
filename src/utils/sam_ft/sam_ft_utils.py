import os

import cv2
import torch
from box import Box
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image #.numpy().transpose(1, 2, 0)
