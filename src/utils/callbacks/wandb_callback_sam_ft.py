import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import lightning as pl
import torch
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid

from src.data.coco.components.coco_dataset import COCODataset
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import random
import math
from src.utils.sam_ft.sam_ft_utils import draw_image
import torch.nn.functional as F

class WandbCallback_SAM_FT(Callback):
    def __init__(
        self,
        data_path: str = "data/coco/images/val2017",
        annotation_path: str = "data/coco/annotations/instances_val2017.json",
        n_images_to_log: int = 4,
        img_size: int = 1024,
    ):
        self.data_path = data_path
        self.annotation_path = annotation_path
        
        self.img_size = img_size

        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []

        self.batch_size = 1
        self.num_samples = n_images_to_log
        self.num_batch = 0
        
        self.dataset = COCODataset(data_dir=data_path,
                          annotation_file=annotation_path,
                          transform=None)
                
        # np.array(Image.open(image_path).convert("RGB"))
        # self.sample_image_height, self.sample_image_width = (
        #     self.sample_image.shape[0],
        #     self.sample_image.shape[1],
        # )
        
        self.image_ids = random.sample(self.dataset.image_ids, self.num_samples)
        

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        predictor = trainer.model.net.get_predictor()
        
        sample_image = []
        for image_id in self.image_ids:
            image_info = self.dataset.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.data_path, image_info['file_name'])
            image = np.array(Image.open(image_path).convert("RGB"))
            
            ann_ids = self.dataset.coco.getAnnIds(imgIds=image_id)
            anns = self.dataset.coco.loadAnns(ann_ids)
            
            bboxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
            
            bboxes = torch.as_tensor(bboxes, device=trainer.model.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
            predictor.set_image(image)
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
            image_output = image_output.unsqueeze(0)
            image_output = F.interpolate(image_output, size=(self.img_size//2, self.image_ids//2), mode='bilinear', align_corners=False)

            image_output = image_output.queeze()
            sample_image.append(image_output)
        
        grid = make_grid(sample_image, nrow=math.sqrt(self.num_samples))
        grid = grid.cpu().numpy().transpose(1, 2, 0)
        
        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="Samples",
            images=[Image.fromarray(grid)],
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):        
        self.visualize(trainer, pl_module, self.image_ids, "Train epoch end")

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        image_ids = random.sample(self.dataset.image_ids, self.num_samples)
        
        self.visualize(trainer, pl_module, image_ids, "Validation epoch end")
        
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        image_ids = random.sample(self.dataset.image_ids, self.num_samples)
        
        self.visualize(trainer, pl_module, image_ids, "Test epoch end")
        
        
    def visualize(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", ids: list[int], tags: str):
        predictor = trainer.model.net.get_predictor()
        
        sample_image = []
        for image_id in ids:
            image_info = self.dataset.coco.loadImgs(image_id)[0]
            image_path = os.path.join(self.data_path, image_info['file_name'])
            image = np.array(Image.open(image_path).convert("RGB"))
            
            ann_ids = self.dataset.coco.getAnnIds(imgIds=image_id)
            anns = self.dataset.coco.loadAnns(ann_ids)
            
            bboxes = []
            for ann in anns:
                x, y, w, h = ann['bbox']
                bboxes.append([x, y, x + w, y + h])
            
            bboxes = torch.as_tensor(bboxes, device=trainer.model.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
            predictor.set_image(image)
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            image_output = draw_image(image, masks.squeeze(1), boxes=None, labels=None)
            image_output = image_output.unsqueeze(0)
            image_output = F.interpolate(image_output, size=(self.img_size//2, self.image_ids//2), mode='bilinear', align_corners=False)

            image_output = image_output.queeze()
            sample_image.append(image_output)
        
        grid = make_grid(sample_image, nrow=math.sqrt(self.num_samples))
        grid = grid.cpu().numpy().transpose(1, 2, 0)
        
        # image = sample_image[0].cpu().numpy().transpose(1, 2, 0)
        
        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key=tags,
            images=[Image.fromarray(grid)],
        )
        