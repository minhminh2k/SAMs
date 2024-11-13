import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou

class IoULoss(nn.Module):

    def __init__(self, tag = "IoU_loss", weight=None, size_average=True):
        super().__init__()
        self.tag = tag

    def forward(self, inputs, targets, iou_predictions, num_masks):
        batch_iou = calc_iou(inputs, targets)
        iou_loss = F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
        return iou_loss