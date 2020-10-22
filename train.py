"""
Author's Note:
--------------
Adapted my notebook constructs to run in a CLI / command-line environment

Some of the code is more poorly factored in this environment, because
1. I wanted to add types
2. My assignment is late enough and I don't want to get charged again for something
   that's done, for cosmetic reasons.

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
from pathlib import Path

from nn_trainer.model_loading import PersistableNet, get_pretrained_model, organize_data
from nn_trainer.training import NeuralNetworkTrainer
from nn_trainer.utils import build_training_arg_parser

import torch
from torch import nn, optim


def configure_model(model: nn.Module, learning_rate: float, device: torch.device):
    """Default training configuration, parameterized for the assignment requirements"""
    loss_function = nn.CrossEntropyLoss()

    configured = model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(parameters, lr=learning_rate)
    learning_rate_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)
    return configured, loss_function, optimizer, learning_rate_scheduler


def train_model(args: Namespace) -> PersistableNet:
    """
    Translate the arguments into function calls
    Construct a NeuralNetworkTrainer
    Construct the limited hyperparameter abstractions to pass to `trainer.train`
    Load a pretrained model
    """
    print("Constructing dataset references")
    image_datasets, dataloaders, dataset_sizes = organize_data(args.data_directory)
    print("Loaded dataset sizes:")
    print(f"\tTrain:  {dataset_sizes.train:3} images")
    print(f"\tvalid: {dataset_sizes.valid:3} images")
    print(f"\ttest:  {dataset_sizes.test:3} images")

    # should the number of classes be the number from the dataset?
    n_classes = len(image_datasets['train'].classes)

    print("Detected classes =", n_classes)
    print("Getting pre-trained model")
    model = get_pretrained_model(args.arch,
                                 n_classes,
                                 additional_hidden_units=int(args.hidden_units or 0),
                                 frozen=True)
    training_device = torch.device('cuda:0' if args.gpu else 'cpu')

    (configured_model, loss_function, training_optimizer,
     training_lr_scheduler) = configure_model(model, args.learning_rate, training_device)

    trainer = NeuralNetworkTrainer(dataloaders.train,
                                   dataloaders.valid,
                                   training_device,
                                   checkpoint_directory=args.save_dir,
                                   dataset_sizes=dataset_sizes)

    trainer.train(configured_model,
                  loss_function,
                  training_optimizer,
                  training_lr_scheduler,
                  epochs=args.epochs)

    net = PersistableNet(
        trainer._best_model,
        training_optimizer,
        {
            "class_to_idx": image_datasets['train'].class_to_idx,
            "idx_to_class": image_datasets['train'].classes  # encoding mapping :: int -> int
        })

    return net


def main():
    parser = build_training_arg_parser()
    args = parser.parse_args()

    persistable_net = train_model(args)
    persistable_net.save(Path(args.save_dir, 'best-model.pt'))


if __name__ == "__main__":
    main()