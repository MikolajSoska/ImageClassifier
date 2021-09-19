import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torchvision.models import resnet50

import classifier.utils as utils
from classifier.data import load_data


class Trainer:
    def __init__(self, path_to_data: str, batch_size: int, validation_ratio: float, learning_rate: float,
                 use_cuda: bool):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio
        self.learning_rate = learning_rate
        self.device = utils.get_device(use_cuda)

    def train(self):
        train_loader, validation_loader, classes_number = load_data(self.path_to_data, self.batch_size,
                                                                    self.validation_ratio)

        model = resnet50(pretrained=False, num_classes=classes_number)
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        running_loss = []
        running_score = []
        train_size = len(train_loader)
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(self.device)
            labels = labels.to(self.device)

            prediction = model(data)
            loss = criterion(prediction, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            score = accuracy_score(torch.argmax(prediction, dim=-1).cpu(), labels.cpu())
            running_score.append(score)

            if i % 50 == 0:
                mean_loss = sum(running_loss) / len(running_loss)
                mean_score = sum(running_score) / len(running_score)
                print(f'Iteration: {i}/{train_size}, Loss: {mean_loss}, Accuracy: {mean_score}')
                running_loss.clear()
