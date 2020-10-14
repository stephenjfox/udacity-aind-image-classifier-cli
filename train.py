"""
Author's Note:
--------------



Requirements as presented:
--------------------------

Training a network
* train.py successfully trains a new network on a dataset of images

Training validation log
* The training loss, validation loss, and validation accuracy are printed out as a network trains

Model architecture
* The training script allows users to choose from at least two different architectures
  available from torchvision.models

Model hyperparameters
* The training script allows users to set hyperparameters for
  * learning rate,
  * number of hidden units, and
  * training epochs

Training with GPU
* The training script allows users to choose training the model on a GPU
"""

from argparse import ArgumentParser, Namespace
from nn_trainer.model_loading import SUPPORTED_ARCHITECTURES


def parse_arguments() -> Namespace:
    parser = ArgumentParser("Image Classifier Training Program",
                            description="Train your own classifier that works on your images!")

    parser.add_argument(
        'data_directory',
        # required=True,
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

    hp_arg_group = parser.add_argument_group('hyperparameters',
                                             'Options for setting hyperparameters')

    hp_arg_group.add_argument(
        '--learning_rate',
        help=
        'The learning rate for the learned layer, pre-trained atop the pre-trained, frozen model',
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

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # train_model(args)
    pass


if __name__ == "__main__":
    main()