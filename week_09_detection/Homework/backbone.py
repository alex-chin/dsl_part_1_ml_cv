import torch
import torch.nn as nn
import timm

class BackboneUnfreeze(nn.Module):
    def __init__(self, model_name="efficientnet_b0", out_indices=(-1, -2, -3), unfreeze_last=0):
        super().__init__()
        # timm.list_models(pretrained=True)
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True, out_indices=out_indices)
        
        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last N layers if unfreeze_last > 0
        if unfreeze_last > 0:
            # Get all layers of the backbone
            layers = list(self.backbone.children())
            # Unfreeze the last N layers
            for layer in layers[-unfreeze_last:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x) 