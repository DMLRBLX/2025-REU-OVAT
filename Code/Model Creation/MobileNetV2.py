import torch
import torch.nn as nn
import timm  # pip install timm

def get_mobilenet_v2(num_classes=3):
    """
    Returns a MobileNetV2 model (not pretrained).
    """
    model = timm.create_model(
        'mobilenetv2_100',
        pretrained=False,
        num_classes=num_classes
    )
    return model

if __name__ == "__main__":
    # Example: create MobileNetV2 for 3 classes
    model = get_mobilenet_v2(num_classes=3)
    print(model)