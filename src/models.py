import torch
from torchvision import models

NAMES = ("mobilenet_v2", "mobilenet_v3_large", "shufflenet_v2_x1_0")


def build_model(name, num_classes=9, pretrained=True):
    weights = {"mobilenet_v2": models.MobileNet_V2_Weights.DEFAULT,
               "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
               "shufflenet_v2_x1_0": models.ShuffleNet_V2_X1_0_Weights.DEFAULT}
    if name not in NAMES:
        raise ValueError(f"Unknown architecture: {name}")
    model = getattr(models, name)(weights=weights[name] if pretrained else None)
    if name == "shufflenet_v2_x1_0":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
