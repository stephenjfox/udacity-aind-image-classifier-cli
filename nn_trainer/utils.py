from argparse import ArgumentParser
from datetime import datetime
from nn_trainer.model_loading import SUPPORTED_ARCHITECTURES

import torch


def accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    return (predictions == labels).float().mean()


def now_timestamp() -> str:
    """Produce a timestamp as a string"""
    time_ = datetime.now()
    stamp = time_.strftime("%Y.%m.%d-%H:%M:%S")
    return stamp


def build_training_arg_parser() -> ArgumentParser:
    parser = ArgumentParser("Image Classifier Training Program",
                            description="Train your own classifier that works on your images!")

    parser.add_argument('data_directory',
                        help='Directory of images you want a model to learn to classify')

    parser.add_argument(
        '--save_dir',
        help='Directory to save checkpoint, intermittent models during training',
        default='model_candidates',
        type=str)

    parser.add_argument(
        '--arch',
        help='Choice of model architecture, pre-trained on the ImageNet-1000 dataset',
        default='resnet18',
        choices=SUPPORTED_ARCHITECTURES)

    parser.add_argument('--gpu', help='Enable GPU-based training', action='store_true')

    hp_arg_group = parser.add_argument_group('hyperparameters',
                                             'Options for setting hyperparameters')

    hp_arg_group.add_argument(
        '--learning_rate',
        help='The learning rate for the learned layer, pre-trained atop the pre-trained, '
        'frozen model',
        default=3e-3,
        type=float)

    # TODO: check if this is the correct understanding of the requirement.
    # Shouldn't the output number of hidden units be the number of
    # distinct classes in the `data_directory?`
    hp_arg_group.add_argument(
        '--hidden_units',
        help='The number of units in an additional hidden layer for the classification',
        type=int)

    hp_arg_group.add_argument(
        '--epochs',
        help='The number of epochs (full passes through the data set) to train with',
        default=5,
        type=int)

    return parser