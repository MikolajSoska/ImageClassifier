from collections import defaultdict
from functools import partial
from typing import Callable

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

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            self.__train_epoch(epoch)

    def __train_epoch(self, epoch: int):
        losses = []
        scores = defaultdict(list)
        train_size = len(self.train_data)
        for i, (data, labels) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            labels: Tensor = labels.to(self.device)

            prediction: Tensor = self.model(data)
            loss = self.criterion(prediction, labels)
            losses.append(loss.item())

            for score, scorer in self.scorers.items():
                scores[score].append(scorer(prediction, labels))

            loss.backward()
            self.optimizer.step()

            if i % 50 == 0:
                mean_loss = sum(losses) / len(losses)
                scores_str = ', '.join(f'{score}: {sum(values) / len(values)}' for score, values in scores.items())
                print(f'Epoch: {epoch}, Iteration: {i}/{train_size}, Loss: {mean_loss}, {scores_str}')
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
