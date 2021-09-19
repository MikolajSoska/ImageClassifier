import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
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

    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(self.epochs):
            self.__train_epoch(epoch)

    def __train_epoch(self, epoch: int):
        running_loss = []
        running_score = []
        train_size = len(self.train_data)
        for i, (data, labels) in enumerate(self.train_data):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)

            prediction = self.model(data)
            loss = self.criterion(prediction, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())
            score = accuracy_score(torch.argmax(prediction, dim=-1).cpu(), labels.cpu())
            running_score.append(score)

            if i % 50 == 0:
                mean_loss = sum(running_loss) / len(running_loss)
                mean_score = sum(running_score) / len(running_score)
                print(f'Epoch: {epoch}, Iteration: {i}/{train_size}, Loss: {mean_loss}, Accuracy: {mean_score}')
                running_loss.clear()
