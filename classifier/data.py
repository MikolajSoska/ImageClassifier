from pathlib import Path
from typing import Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets import ImageFolder


class TestDataset(Dataset):
    """
    Custom Dataset class for storing testing images with their identifiers (file names)
    """

    def __init__(self, path_to_images: str):
        """
        Initialize TestDataset class. Images are resized to 256x256 size and normalized with mean=0.5, std=0.5 values.
        :param path_to_images: path to directory with images in JPG format
        """
        self.images = [image for image in Path(path_to_images).glob('*.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, index: int) -> T_co:
        """
        Method loads given image and perform transformations.
        :param index: Index of image to open.
        :return: Tuple with Transformed image in Tensor object and image filename
        """
        image_path = self.images[index]
        image = Image.open(image_path)
        image = self.transform(image)
        image_id = f'{image_path.stem}.jpg'

        return image, image_id

    def __len__(self) -> int:
        """
        Method returns dataset size.
        :return: dataset size
        """
        return len(self.images)


def load_data(path_to_images: str, batch_size: int, validation_ratio: float = None,
              random_seed: int = None) -> Union[Tuple[DataLoader, int], Tuple[DataLoader, DataLoader, int]]:
    """
    Method loads images from specified path. Images are loaded with ImageFolder class, so they need to be stored in
    format:
    - root_dir:
        - class_1: a.jpg, b.jpg
        - class_2: c.jpg, d.jpg
        ...

    Each image is passed through sequence of predefined transformation: random cropping, horizontal flipping and
    rotation. Images are also normalized with mean=0.5 and std=0.5. Each image is outputted in 256x256 size.

    Data can be optionally split into training and validation data in stratified fashion, if appropriate parameters are
    specified.
    :param path_to_images: path to directory with images.
    :param batch_size: batch size used in Dataloader
    :param validation_ratio: optional argument, if passes, method will take this percent of images into validation set
    :param random_seed: optional argument, used as random_state during train/val split
    :return: Tuple with training Dataloader and number of classes, or tuple with training Dataloader,
        validation Dataloader and number of classes if `validation_ratio` is not equal to None
    """
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
                                                             stratify=dataset.targets, random_state=random_seed)
        train_dataset = Subset(dataset, train_indices)
        validation_dataset = Subset(dataset, validation_indices)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
        return train_loader, validation_loader, classes_number
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), classes_number


def load_test_data(path_to_images: str, batch_size: int) -> DataLoader:
    """
    Method loads testing data with TestDataset class.
    :param path_to_images: path to directory with images.
    :param batch_size: batch size used in Dataloader
    :return: DataLoader with test images
    """
    dataset = TestDataset(path_to_images)
    dataloader = DataLoader(dataset, batch_size)

    return dataloader
