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

from argparse import Namespace
from nn_trainer.utils import build_arg_parser
from nn_trainer import NeuralNetworkTrainer
from nn_trainer.model_loading import organize_data


def train_model(args: Namespace):
    """
    Translate the arguments into function calls
    Construct a NeuralNetworkTrainer
    Construct the limited hyperparameter abstractions to pass to trainer.train()
    """
    (training_dataloader,
     validation_dataloader), dataset_sizes = organize_data(args.data_directory)
    trainer = NeuralNetworkTrainer(training_dataloader,
                                   validation_dataloader,
                                   'cuda:0' if args.gpu else 'cpu',
                                   checkpoint_directory=args.save_dir,
                                   dataset_sizes=dataset_sizes)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()