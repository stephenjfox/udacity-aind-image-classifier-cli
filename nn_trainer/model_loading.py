from torchvision import models

_architectures = {
'alexnet': models.alexnet,
'resnet18': models.resnet18,
'resnet34': models.resnet34,
'vgg11': models.vgg11,
}

SUPPORTED_ARCHITECTURES = list(_architectures.keys())