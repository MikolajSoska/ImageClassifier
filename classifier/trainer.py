import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, List, Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import classifier.utils as utils


class Trainer:
    """
    Trainer class for neural network classifier training and validation.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, train_data: DataLoader,
                 val_data: DataLoader, epochs: int, use_cuda: bool, log_directory: str, save_directory: str,
                 run_id: str):
        """
        Initialize Trainer class.
        :param model: Neural network model to train
        :param criterion: loss function used during training and validation
        :param optimizer: optimizer used during training
        :param train_data: data used during training
        :param val_data: data used during validation
        :param epochs: number of epochs to train model
        :param use_cuda: specify if GPU should be used during training
        :param log_directory: directory where Tensorboard logs will be saved
        :param save_directory: directory where model checkpoint will be saved
        :param run_id: Unique string that will identify this run in logs and checkpoints
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.device = utils.get_device(use_cuda)
        self.save_directory = Path(save_directory)
        self.run_id = run_id
        self.scorers = {
            'Accuracy': self.__score_accuracy,
            'Precision': self.__score_precision,
            'Recall': self.__score_recall,
            'F1': self.__score_f1
        }
        self.train_size = len(train_data)
        self.val_size = len(val_data)

        images, _ = next(iter(train_data))
        self.tensorboard = SummaryWriter(f'{log_directory}/{run_id}')
        self.tensorboard.add_graph(self.model, images)

    def train(self, verbosity: int = 50) -> None:
        """
        Method runs model training. Validation is performed after each completed epoch.
        :param verbosity: Number of iterations between training messages
        """
        self.model = self.model.to(self.device)
        self.model.train()
        checkpoint_epoch = self.__load_checkpoint()
        for epoch in range(checkpoint_epoch + 1, self.epochs):
            self.__train_epoch(epoch, verbosity)
            self.__save_checkpoint(epoch)
            self.__validate_model(epoch)

    def __train_epoch(self, epoch: int, verbosity: int) -> None:
        """
        Training loop for one epoch.
        :param epoch: Current epoch number
        :param verbosity: Number of iterations between training messages
        """
        losses = []
        scores = defaultdict(list)
        time_start = time.time()
        for i, (data, labels) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            loss = self.__model_step(data, labels, losses, scores)
            loss.backward()
            self.optimizer.step()

            if i % verbosity == 0:
                self.__log_performance(epoch, i, self.train_size, losses, scores, time_start, 'train')
                time_start = time.time()

    def __validate_model(self, epoch: int) -> None:
        """
        Method evaluates model with validation data.
        :param epoch: Current epoch number
        """
        self.model.eval()
        losses = []
        scores = defaultdict(list)
        print('Starting validation phase...')
        with torch.no_grad():
            time_start = time.time()
            for i, (data, labels) in enumerate(tqdm(self.val_data, desc='Validating data', total=self.val_size,
                                                    file=sys.stdout)):
                self.__model_step(data, labels, losses, scores)

        print('Validation results:')
        self.__log_performance(epoch, self.val_size, self.val_size, losses, scores, time_start, 'validation')
        self.model.train()

    def __model_step(self, data: Tensor, labels: Tensor, losses: List[float], scores: Dict[str, List[float]]) -> Tensor:
        """
        Method passes single batch of data into model and computes loss and scores.
        after prediction.
        :param data: Data batch
        :param labels: Data real labels
        :param losses: List with previous loss values
        :param scores: Dictionary with previous scores values
        :return: Loss tensor used in backward pass
        """
        data = data.to(self.device)
        labels = labels.to(self.device)

        prediction = self.model(data)
        loss = self.criterion(prediction, labels)
        losses.append(loss.item())

        for score, scorer in self.scorers.items():
            value = scorer(prediction, labels)
            scores[score].append(value)

        return loss

    def __save_checkpoint(self, epoch: int) -> None:
        """
        Method saves current state of model and optimizer to checkpoint file.
        :param epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        self.save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.save_directory / f'model-{self.run_id}.pt')

    def __load_checkpoint(self) -> int:
        """
        Method loads saved state of model and optimizer from checkpoint file.
        :return: Loaded epoch number
        """
        checkpoint_path = self.save_directory / f'model-{self.run_id}.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f'Loaded checkpoint from: {checkpoint_path}')
        else:
            epoch = -1
        return epoch

    def __log_performance(self, epoch: int, iteration: int, iteration_max: int, losses: List[float],
                          scores: Dict[str, List[float]], time_start: float, phase: str) -> None:
        """
        Method logs curring training/validation performance
        :param epoch: Current epoch number
        :param iteration: Current iteration number
        :param iteration_max: Maximum number of iteration in one epoch
        :param losses: List with previous losses
        :param scores: Dictionary with previous scores
        :param time_start: Timestamp of training segment
        :param phase: Current Trainer phase (e.g. "train", "validation")
        """
        iteration_time = time.time() - time_start
        mean_loss = sum(losses) / len(losses)
        iteration_step = epoch * iteration_max + iteration

        score_values = []
        for score, values in scores.items():
            value = sum(values) / len(values)
            score_values.append(f'{score}: {value}')
            self.tensorboard.add_scalar(f'{score}/{phase}', value, iteration_step)
        self.tensorboard.add_scalar(f'Loss/{phase}', mean_loss, iteration_step)

        scores_str = ', '.join(score_values)
        print(f'Epoch: {epoch}, Iteration: {iteration}/{iteration_max}, Time: {iteration_time}, Loss: {mean_loss}, '
              f'{scores_str}')
        losses.clear()
        scores.clear()

    @staticmethod
    def __score_classification(prediction: Tensor, target: Tensor, scorer: Callable) -> float:
        """
        Method calculates given classification score.
        :param prediction: Tensor with predicted values for each class
        :param target: Tensor with target labels
        :param scorer: Scorer method
        :return: classification score
        """
        labels = torch.argmax(prediction, dim=-1)
        return scorer(labels.cpu(), target.cpu())

    @staticmethod
    def __score_accuracy(prediction: Tensor, target: Tensor) -> float:
        """
        Method calculates accuracy score.
        :param prediction: Tensor with predicted values for each class
        :param target: Tensor with target labels
        :return: classification score
        """
        return Trainer.__score_classification(prediction, target, accuracy_score)

    @staticmethod
    def __score_precision(prediction: Tensor, target: Tensor) -> float:
        """
        Method calculates precision score.
        :param prediction: Tensor with predicted values for each class
        :param target: Tensor with target labels
        :return: classification score
        """
        return Trainer.__score_classification(prediction, target, partial(precision_score, average='micro'))

    @staticmethod
    def __score_recall(prediction: Tensor, target: Tensor) -> float:
        """
        Method calculates recall score.
        :param prediction: Tensor with predicted values for each class
        :param target: Tensor with target labels
        :return: classification score
        """
        return Trainer.__score_classification(prediction, target, partial(recall_score, average='micro'))

    @staticmethod
    def __score_f1(prediction: Tensor, target: Tensor) -> float:
        """
        Method calculates F1 score.
        :param prediction: Tensor with predicted values for each class
        :param target: Tensor with target labels
        :return: classification score
        """
        return Trainer.__score_classification(prediction, target, partial(f1_score, average='micro'))
