from typing import Tuple, Union

import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder


def load_data(path_to_images: str, batch_size: int,
              validation_ratio: float = None) -> Union[Tuple[DataLoader, int], Tuple[DataLoader, DataLoader, int]]:
    transform = transforms.Compose([
        transforms.RandomCrop(256, pad_if_needed=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = ImageFolder(path_to_images, transform=transform)
    classes_number = len(set(dataset.targets))
    if validation_ratio is not None:

        train_indices, validation_indices = train_test_split(torch.arange(len(dataset)), test_size=validation_ratio,
                                                             stratify=dataset.targets)
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
        return train_loader, validation_loader, classes_number
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), classes_number
