import torch
from torchvision import models

resNet50 = models.resnet50(pretrained=True)
vits_8 = models.convnext_small(pretrained=True)

print(resNet50)