import torch
import torch.nn as nn
from typing import List, Dict, Optional, Union

class FPN(nn.Module):
    """
    Feature Pyramid Network neck implementation.
    Takes features from backbone and creates a pyramid of features with different scales.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        extra_blocks: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Create lateral convolution to reduce channels
        self.lateral_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        # Create output convolution to smooth features
        self.output_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
            
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.extra_blocks = extra_blocks

    def forward(self, features: Union[List[torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Either a list of feature maps from backbone, ordered from highest to lowest resolution,
                     or a dictionary of feature maps with keys indicating their resolution level
        Returns:
            Dict[str, torch.Tensor]: Dictionary of feature maps with different scales
        """
        # Convert dictionary to list if needed
        if isinstance(features, dict):
            features = [features[f] for f in sorted(features.keys())]
            
        # Process features from backbone
        laterals = []
        for feature in features:
            laterals.append(self.lateral_conv(feature))
            
        # Top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample and add
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], scale_factor=2, mode="nearest"
            )
            
        # Apply output convolutions
        outs = []
        for lateral in laterals:
            outs.append(self.output_conv(lateral))
            
        # Add extra levels if needed
        if self.extra_blocks is not None:
            outs = self.extra_blocks(outs)
            
        # Convert to dictionary format
        return {f"P{i}": out for i, out in enumerate(outs)} 