import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

def compute_task_alignment_score(pred_boxes, target_boxes, pred_cls, target_cls, alpha=1.0, beta=6.0):
    """
    Compute task alignment score between predictions and targets.
    
    Args:
        pred_boxes: Predicted boxes in format (x_min, y_min, w, h)
        target_boxes: Target boxes in format (x_min, y_min, w, h)
        pred_cls: Predicted class probabilities
        target_cls: Target class labels (one-hot encoded)
        alpha: Weight for classification score
        beta: Weight for localization score
    
    Returns:
        Task alignment scores
    """
    # Convert boxes to (x_min, y_min, x_max, y_max) format
    pred_x1 = pred_boxes[..., 0]
    pred_y1 = pred_boxes[..., 1]
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2]
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3]
    
    target_x1 = target_boxes[..., 0]
    target_y1 = target_boxes[..., 1]
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2]
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3]
    
    # Calculate IoU
    pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
    target_boxes_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=-1)
    iou = box_iou(pred_boxes_xyxy, target_boxes_xyxy)
    
    # Calculate classification score
    cls_score = torch.sum(pred_cls * target_cls, dim=-1)
    
    # Calculate task alignment score
    alignment_score = (cls_score ** alpha) * (iou ** beta)
    
    return alignment_score

def assign_target_tal(anchors, gt_boxes, gt_labels, num_classes, pred_boxes=None, pred_cls=None, 
                     pos_th=0.6, neg_th=0.3, alpha=1.0, beta=6.0):
    """
    Assign targets using Task Alignment Learning (TAL) from TOOD.
    
    Args:
        anchors: Anchor boxes in format (x_min, y_min, x_max, y_max)
        gt_boxes: Ground truth boxes in format (x_min, y_min, w, h)
        gt_labels: Ground truth class labels
        num_classes: Number of classes
        pred_boxes: Predicted boxes (if available)
        pred_cls: Predicted class probabilities (if available)
        pos_th: Positive threshold for IoU
        neg_th: Negative threshold for IoU
        alpha: Weight for classification score in TAL
        beta: Weight for localization score in TAL
    
    Returns:
        target_offsets: Target offsets for positive anchors
        target_objectness: Objectness targets (1: positive, 0: negative, -1: ignore)
        target_cls: Classification targets
    """
    num_anchors = anchors.shape[0]
    target_objectness = torch.zeros(num_anchors, device=anchors.device)
    target_offsets = torch.zeros((num_anchors, 4), device=anchors.device)
    target_cls = torch.zeros((num_anchors, num_classes), device=anchors.device)
    
    if gt_boxes.numel() == 0:
        return target_offsets, target_objectness, target_cls
    
    # Convert GT boxes to (x_min, y_min, x_max, y_max) format
    gt_xyxy = gt_boxes.clone()
    gt_xyxy[:, 2:] = gt_xyxy[:, :2] + gt_xyxy[:, 2:]
    
    # Calculate IoU between anchors and GT boxes
    ious = box_iou(anchors, gt_xyxy)
    
    # If we have predictions, use TAL to compute alignment scores
    if pred_boxes is not None and pred_cls is not None:
        alignment_scores = compute_task_alignment_score(
            pred_boxes, gt_boxes, pred_cls, F.one_hot(gt_labels, num_classes),
            alpha=alpha, beta=beta
        )
        # Use alignment scores instead of IoU for assignment
        best_score, best_gt_idx = alignment_scores.max(dim=1)
    else:
        # Fallback to IoU-based assignment
        best_iou, best_gt_idx = ious.max(dim=1)
        best_score = best_iou
    
    # Assign negative samples
    neg_mask = best_score < neg_th
    target_objectness[neg_mask] = 0
    
    # Assign positive samples
    pos_mask = best_score >= pos_th
    target_objectness[pos_mask] = 1
    
    # Assign ignore samples
    ignore_mask = (best_score >= neg_th) & (best_score < pos_th)
    target_objectness[ignore_mask] = -1
    
    # Assign regression targets for positive samples
    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    for pos in pos_indices:
        gt_idx = best_gt_idx[pos]
        gt_box = gt_xyxy[gt_idx]
        anchor_box = anchors[pos]
        
        # Calculate target offsets
        target_offsets[pos] = get_target_offset(anchor_box, gt_box)
        target_cls[pos, gt_labels[gt_idx]] = 1
    
    # Ensure each GT has at least one positive anchor
    for gt_idx in range(gt_xyxy.shape[0]):
        if not ((target_objectness == 1) & (best_gt_idx == gt_idx)).any():
            # Find anchor with highest IoU for this GT
            best_anchor_idx = torch.argmax(ious[:, gt_idx])
            target_offsets[best_anchor_idx] = get_target_offset(anchors[best_anchor_idx], gt_xyxy[gt_idx])
            target_objectness[best_anchor_idx] = 1
            target_cls[best_anchor_idx, gt_labels[gt_idx]] = 1
    
    return target_offsets, target_objectness, target_cls

def get_target_offset(anchor_box, gt_box):
    """
    Calculate target offsets between anchor and ground truth box.
    
    Args:
        anchor_box: Anchor box in format (x_min, y_min, x_max, y_max)
        gt_box: Ground truth box in format (x_min, y_min, x_max, y_max)
    
    Returns:
        Target offsets (tx, ty, tw, th)
    """
    # Convert to center, width, height format
    anchor_center_x = (anchor_box[0] + anchor_box[2]) / 2
    anchor_center_y = (anchor_box[1] + anchor_box[3]) / 2
    anchor_width = anchor_box[2] - anchor_box[0]
    anchor_height = anchor_box[3] - anchor_box[1]
    
    gt_center_x = (gt_box[0] + gt_box[2]) / 2
    gt_center_y = (gt_box[1] + gt_box[3]) / 2
    gt_width = gt_box[2] - gt_box[0]
    gt_height = gt_box[3] - gt_box[1]
    
    # Calculate offsets
    tx = (gt_center_x - anchor_center_x) / anchor_width
    ty = (gt_center_y - anchor_center_y) / anchor_height
    tw = torch.log(gt_width / anchor_width)
    th = torch.log(gt_height / anchor_height)
    
    return torch.tensor([tx, ty, tw, th], device=anchor_box.device) 