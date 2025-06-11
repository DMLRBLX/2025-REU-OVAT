import torch
import torch.nn as nn
import timm  # pip install timm

def get_resnet34(num_classes=3):
    """
    Returns a ResNet34 model (not pretrained).
    """
    model = timm.create_model(
        'resnet34',
        pretrained=False,
        num_classes=num_classes
    )
    return model

if __name__ == "__main__":
    # Example: create ResNet34 for 3 classes
    model = get_resnet34(num_classes=3)
    print(model)