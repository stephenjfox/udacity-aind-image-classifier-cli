import os
from typing import Callable, Dict, List, NamedTuple

from nn_trainer.types import DatasetSizes

from torch import nn
from torch.tensor import Tensor
from torch.utils.data import DataLoader, Dataset
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


class TaggedDataLoaders(NamedTuple):
    train: DataLoader
    valid: DataLoader
    test: DataLoader


class DataLoadingComponents(NamedTuple):
    image_datasets: Dict[str, Dataset]
    dataloaders: TaggedDataLoaders
    dataset_sizes: DatasetSizes


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


def get_pretrained_model(model_name: str,
                         num_classes: int,
                         additional_hidden_units: int = 0,
                         frozen=False) -> nn.Module:
    """
    Helper to prevent duplicating highly stateful code
    """
    model_constructor = _architectures[model_name]
    model = model_constructor(pretrained=True)
    if frozen:
        # turn off gradient computation
        for param in model.parameters():
            param.requires_grad = False

    n_last_layers_input_features = model.fc.in_features

    # TO reviewer: is this what you want? The instructions are unclear
    inference_head: nn.Module
    if additional_hidden_units:
        hidden_layer = nn.Linear(n_last_layers_input_features, additional_hidden_units)
        output_layer = nn.Linear(additional_hidden_units, num_classes)

        inference_head = nn.Sequential(hidden_layer, output_layer)
    else:
        inference_head = nn.Linear(n_last_layers_input_features, num_classes)

    model.fc = inference_head

    return model


def organize_data(
        data_dir: str,
        named_transforms: List[NamedTransform] = DEFAULT_TRANSFORMS) -> DataLoadingComponents:
    """
    Load data from `data_dir`, applying transformations where the subdirectory
    matches the `NamedTransform.name` of the instances iof `named_transforms`
    """

    image_datasets = {
        ds_name: datasets.ImageFolder(os.path.join(data_dir, ds_name), tfms)
        for ds_name, tfms in named_transforms
    }
    loaders_dict = {
        ds_name: DataLoader(img_dataset, batch_size=16, shuffle=True, num_workers=1)
        for ds_name, img_dataset in image_datasets.items()
    }
    dataloaders = TaggedDataLoaders(loaders_dict['train'], loaders_dict['valid'],
                                    loaders_dict['test'])

    sizes = DatasetSizes(len(image_datasets['train']), len(image_datasets['valid']),
                         len(image_datasets['test']))

    return DataLoadingComponents(image_datasets, dataloaders, sizes)
