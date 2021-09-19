import time
from collections import defaultdict
from functools import partial
from typing import Callable, List, Dict

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

import classifier.utils as utils


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer, train_data: DataLoader,
                 val_data: DataLoader, epochs: int, use_cuda: bool):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.device = utils.get_device(use_cuda)
        self.scorers = {
            'Accuracy': self.__score_accuracy,
            'Precision': self.__score_precision,
            'Recall': self.__score_recall,
            'F1': self.__score_f1
        }
        self.train_size = len(train_data)
        self.val_size = len(val_data)

    def train(self, verbosity: int = 50) -> None:
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            self.__train_epoch(epoch, verbosity)
            self.__validate_model(epoch)

    def __train_epoch(self, epoch: int, verbosity: int) -> None:
        losses = []
        scores = defaultdict(list)
        time_start = time.time()
        for i, (data, labels) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            loss = self.__model_step(data, labels, losses, scores)
            loss.backward()
            self.optimizer.step()

            if i % verbosity == 0:
                self.__log_performance(epoch, i, self.train_size, losses, scores, time_start)
                time_start = time.time()

    def __validate_model(self, epoch: int) -> None:
        self.model.eval()
        losses = []
        scores = defaultdict(list)
        print('Starting validation phase...')
        with torch.no_grad():
            time_start = time.time()
            for data, labels in self.val_data:
                self.__model_step(data, labels, losses, scores)

        self.__log_performance(epoch, self.val_size, self.val_size, losses, scores, time_start)
        self.model.train()

    def __model_step(self, data: Tensor, labels: Tensor, losses: List[float], scores: Dict[str, List[float]]) -> Tensor:
        data = data.to(self.device)
        labels = labels.to(self.device)

        prediction = self.model(data)
        loss = self.criterion(prediction, labels)
        losses.append(loss.item())

        for score, scorer in self.scorers.items():
            scores[score].append(scorer(prediction, labels))

        return loss

    @staticmethod
    def __log_performance(epoch: int, iteration: int, iteration_max: int, losses: List[float],
                          scores: Dict[str, List[float]], time_start: float) -> None:
        iteration_time = time.time() - time_start
        mean_loss = sum(losses) / len(losses)
        scores_str = ', '.join(f'{score}: {sum(values) / len(values)}' for score, values in scores.items())
        print(f'Epoch: {epoch}, Iteration: {iteration}/{iteration_max}, Time: {iteration_time}, Loss: {mean_loss}, '
              f'{scores_str}')
        losses.clear()
        scores.clear()

    @staticmethod
    def __score_classification(prediction: Tensor, target: Tensor, scorer: Callable) -> float:
        labels = torch.argmax(prediction, dim=-1)
        return scorer(labels, target)

    @staticmethod
    def __score_accuracy(prediction: Tensor, target: Tensor) -> float:
        return Trainer.__score_classification(prediction, target, accuracy_score)

    @staticmethod
    def __score_precision(prediction: Tensor, target: Tensor) -> float:
        return Trainer.__score_classification(prediction, target, partial(precision_score, average='micro'))

    @staticmethod
    def __score_recall(prediction: Tensor, target: Tensor) -> float:
        return Trainer.__score_classification(prediction, target, partial(recall_score, average='micro'))

    @staticmethod
    def __score_f1(prediction: Tensor, target: Tensor) -> float:
        return Trainer.__score_classification(prediction, target, partial(f1_score, average='micro'))
