import os
from typing import Callable, List, NamedTuple
from torch.tensor import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

_architectures = {
    'alexnet': models.alexnet,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'vgg11': models.vgg11,
}

SUPPORTED_ARCHITECTURES = list(_architectures.keys())

PRETRAINED = {
    'means': [0.485, 0.456, 0.406],
    'stddevs': [0.229, 0.224, 0.225],
}


class NamedTransform(NamedTuple):
    name: str
    transform: Callable[['Image'], Tensor]


DEFAULT_TRANSFORMS = [
    NamedTransform(
        'train',
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(PRETRAINED['means'], PRETRAINED['stddevs'])
        ])),
    NamedTransform(
        'valid',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(PRETRAINED['means'], PRETRAINED['stddevs'])
        ])),
    NamedTransform(
        'test',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(PRETRAINED['means'], PRETRAINED['stddevs'])
        ]))
]


def organize_data(data_dir: str, named_transforms: List[NamedTransform] = DEFAULT_TRANSFORMS):
    """
    Load data from `data_dir`, applying transformations where the subdirectory
    matches the `NamedTransform.name` of the instances iof `named_transforms`
    """

    image_datasets = {
        ds_name: datasets.ImageFolder(os.path.join(data_dir, ds_name), tfms)
        for ds_name, tfms in named_transforms
    }
    dataloaders = {
        ds_name: DataLoader(img_dataset, batch_size=16, shuffle=True, num_workers=1)
        for ds_name, img_dataset in image_datasets.items()
    }
    dataset_sizes = {
        ds_name: len(img_dataset)
        for ds_name, img_dataset in image_datasets.items()
    }

    # TODO: return something useful. Do you still need dataset_sizes?
    return dataloaders, dataset_sizes
