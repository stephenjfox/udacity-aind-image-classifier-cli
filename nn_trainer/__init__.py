import copy, time
from nn_trainer.utils import accuracy, now_timestamp
from typing import Tuple, TypedDict

import torch
from torch import nn, optim


class DatasetSizes(TypedDict):
    train: int
    valid: int
    test: int


# TODO: move to another file. Shouldn't be in __init__, I think
class NeuralNetworkTrainer:
    @staticmethod
    def validate_dataset_sizes(sizes: DatasetSizes):
        assert sizes['train'] > 0, 'Invalid training set size'
        assert sizes['valid'] > 0, 'Invalid validation set size'

    def __init__(self, train_dataloader: torch.utils.data.DataLoader,
                 eval_dataloader: torch.utils.data.DataLoader, device,
                 checkpoint_directory: str, dataset_sizes: DatasetSizes):
        self.validate_dataset_sizes(dataset_sizes)
        self.dataset_sizes = dataset_sizes
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.every = 50
        self.checkpoint_directory = checkpoint_directory

    @staticmethod
    def dataloader_on_device(dl, device):
        for X, y in dl:
            X = X.to(device)
            y = y.to(device)
            yield X, y

    def present_run_metrics(self, phase: str, total_loss,
                            total_correct_preds) -> Tuple[float, float]:
        epoch_loss = total_loss / self.dataset_sizes[phase]
        epoch_accuracy = total_correct_preds / self.dataset_sizes[phase]

        print(f"{phase:6} | Loss: {epoch_loss:3.5f} | Acc: {epoch_accuracy:0.6f}")
        return epoch_loss, epoch_accuracy

    def _training_pass(self, model: nn.Module, loss_fun: nn.Module,
                       opt: optim.Optimizer) -> Tuple[float, int]:
        """
        Does a training epoch, returning (loss, num_correct).
        Tracks and updates the gradients, steps the optimizer.
        """
        print("Training pass")
        model.train()

        total_loss = 0.0
        total_correct_preds = 0
        for i, (inputs, labels) in enumerate(
                self.dataloader_on_device(self.train_dataloader, self.device)):
            activations = model(inputs)
            preds = torch.argmax(activations, 1)
            loss = loss_fun(activations, labels)

            total_loss += loss.item() * inputs.size(0)
            num_correct = torch.sum(preds == labels)
            if num_correct:
                total_correct_preds += num_correct

            loss.backward()  # compute those gradients
            opt.step()  # update the world
            opt.zero_grad()  # prepare for another pass

            if i % self.every == 0:
                print("\tAccuracy:", accuracy(preds, labels))

        return total_loss, total_correct_preds.item()

    def _eval_pass(self, model: nn.Module, loss_fun) -> Tuple[float, int]:
        """
        Does an evaluation epoch, updating no gradients but still measure and return (loss, num_correct)
        """
        print("Evaluation pass")
        model.eval()

        total_loss = 0.0
        total_correct_preds = 0
        for i, (inputs, labels) in enumerate(
                self.dataloader_on_device(self.eval_dataloader, self.device)):
            activations = model(inputs)
            preds = torch.argmax(activations, 1)

            loss = loss_fun(activations, labels)

            total_loss += loss.item() * inputs.size(0)
            num_correct = torch.sum(preds == labels)
            if num_correct:
                total_correct_preds += num_correct

            if i % self.every == 0:
                print("\tAccuracy:", accuracy(preds, labels))

        return total_loss, total_correct_preds.item()

    def train(self, model: nn.Module, loss_fun, optimizer, learning_rate_scheduler, epochs):
        """
        Iterate between train and eval modes of the neural network student
        """
        start_time = time.time()

        best_model_weights = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        best_loss = 500  # SENTINEL

        for epoch in range(epochs):
            learning_rate_scheduler.step()
            print(f"Epoch {epoch:2}/{epochs:2}")
            train_agg_loss, train_num_correct = self._training_pass(model, loss_fun, optimizer)
            self.present_run_metrics("train", train_agg_loss, train_num_correct)

            # validation on the improvements from this epoch
            eval_agg_loss, eval_num_correct = self._eval_pass(model, loss_fun)
            eval_epoch_loss, eval_epoch_accuracy = self.present_run_metrics(
                'valid', eval_agg_loss, eval_num_correct)

            print("Eval vs Best:", eval_epoch_accuracy, 'vs', best_accuracy)

            if (eval_epoch_accuracy > best_accuracy) and (eval_epoch_loss < best_loss):
                print("Doing model weight update")
                best_accuracy = eval_epoch_accuracy
                best_loss = eval_epoch_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(model,
                           f"{self.checkpoint_directory}/best-model_{now_timestamp()}.pt")

        time_elapsed = time.time() - start_time
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        print(f"Best validation accuracy: {best_accuracy:4f}")

        # save the model on the trainer instance
        self._best_model_weights = best_model_weights
        model.load_state_dict(best_model_weights)
        self._best_model = model

    def evaluate(self, test_dataloader: torch.utils.data.DataLoader, loss_fun):
        """Runs validation on the test set"""
        assert hasattr(self, '_best_model'), """
        Attempting to evaluate model before training (setting self._best_model)"""

        model = self._best_model
        model.eval()

        total_loss = 0.0
        total_correct_preds = 0
        for i, (inputs,
                labels) in enumerate(self.dataloader_on_device(test_dataloader, self.device)):
            activations = model(inputs)
            preds = torch.argmax(activations, 1)

            loss = loss_fun(activations, labels)

            total_loss += loss.item() * inputs.size(0)
            num_correct = torch.sum(preds == labels).item()
            if num_correct:
                total_correct_preds += num_correct

            if i % self.every == 0:
                print("\tAccuracy:", accuracy(preds, labels))

        acc = total_correct_preds / self.dataset_sizes['test']
        loss = total_loss / self.dataset_sizes['test']
        print(f"""Overall Statistics:
        ----------------------
        Accuracy: {acc}
        Loss:     {loss}""")