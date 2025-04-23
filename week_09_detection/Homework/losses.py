import torch
import torch.nn as nn
import torch.nn.functional as F

class DIoULoss(nn.Module):
    """Distance Intersection over Union (DIoU) Loss.
    
    DIoU loss considers both the overlap area and the distance between the centers
    of the predicted and ground truth boxes. This helps to better optimize the
    box positions during training.
    
    The loss is computed as:
    DIoU = IoU - (d^2 / c^2)
    where:
    - d is the distance between the centers of the boxes
    - c is the diagonal length of the smallest enclosing box
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: Predicted boxes in format (x_min, y_min, w, h)
            target_boxes: Target boxes in format (x_min, y_min, w, h)
        Returns:
            DIoU loss
        """
        # Convert to (x_min, y_min, x_max, y_max) format
        pred_x1 = pred_boxes[..., 0]
        pred_y1 = pred_boxes[..., 1]
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2]
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3]
        
        target_x1 = target_boxes[..., 0]
        target_y1 = target_boxes[..., 1]
        target_x2 = target_boxes[..., 0] + target_boxes[..., 2]
        target_y2 = target_boxes[..., 1] + target_boxes[..., 3]
        
        # Calculate intersection area
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union area
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-16)
        
        # Calculate center distance
        pred_center_x = (pred_x1 + pred_x2) / 2
        pred_center_y = (pred_y1 + pred_y2) / 2
        target_center_x = (target_x1 + target_x2) / 2
        target_center_y = (target_y1 + target_y2) / 2
        
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Calculate diagonal length of the smallest enclosing box
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        enclose_diagonal = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # Calculate DIoU
        diou = iou - (center_distance / (enclose_diagonal + 1e-16))
        
        # Calculate loss
        loss = 1 - diou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 