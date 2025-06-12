import torch
import torch.nn as nn
import timm  # pip install timm

def get_vit_tiny(num_classes=3):
    """
    Returns a ViT-Tiny model (not pretrained).
    """
    model = timm.create_model(
        'vit_tiny_patch16_224',
        pretrained=False,
        num_classes=num_classes
    )
    return model

if __name__ == "__main__":
    # Example: create ViT-Tiny for 10 classes
    model = get_vit_tiny(num_classes=3)
    print(model)