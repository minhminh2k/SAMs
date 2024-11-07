import torch.nn as nn
import torch.nn.functional as F
from ..original_sam.build_sam import sam_model_registry
from ..original_sam.predictor import SamPredictor

# Fine-tuning SAM using bounding boxes as prompts: https://github.com/luca-medeiros/lightning-sam/tree/main
class Model(nn.Module):

    def __init__(
        self, 
        cfg,
        model_type: str = "vit_b", 
        mode_checkpoint: str = "checkpoints/sam_vit_b_01ec64.pth",
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_type = model_type
        self.model_checkpoint = mode_checkpoint
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.freeze_mask_decoder = freeze_mask_decoder

        self.model = sam_model_registry[self.model_type](checkpoint=self.model_checkpoint)
        self.model.train()
        if self.freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []
        for embedding, bbox in zip(image_embeddings, bboxes):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)