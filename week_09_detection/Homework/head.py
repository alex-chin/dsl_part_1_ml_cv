import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledHead(nn.Module):
    """ Decoupled Head from YOLOX.
    
    The head consists of two separate branches:
    1. Classification branch with multiple conv layers
    2. Regression branch with multiple conv layers
    
    Each branch has its own set of convolutional layers with batch normalization and activation.
    """
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Shared initial conv layer
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=1)
        )
        
        # Regression branch
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 5, kernel_size=1)  # 4 for bbox + 1 for confidence
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Feature tensor from FPN
        Returns:
            tuple: (cls_logits, bbox_preds)
        """
        x = self.shared_conv(x)
        cls_logits = self.cls_branch(x)
        bbox_preds = self.reg_branch(x)
        return cls_logits, bbox_preds 