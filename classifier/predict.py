import argparse
import json
import sys
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm

from classifier.data import load_test_data
from classifier.utils import get_device


def parse_args() -> argparse.Namespace:
    """
    Method parses path arguments.
    :return: Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(description='Parameters for predicting images classes.')
    parser.add_argument('--images-path', type=str, required=True, help='Path to images directory')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--use-gpu', action='store_true', help='Train with CUDA')
    parser.add_argument('--model-path', type=str, default='./saved/classifier.pt', help='Path to trained model')
    parser.add_argument('--config-path', type=str, default='./saved/config.json', help='Path to training config')
    parser.add_argument('--save-path', type=str, default='./saved/output.csv', help='Path to training config')

    return parser.parse_args()


def read_config(path_to_config: str) -> Dict[str, Any]:
    """
    Method opens JSON config file with labels data
    :param path_to_config: Path to config file in JSON format
    :return: Opened config file
    """
    with open(path_to_config, 'r') as config_file:
        config = json.load(config_file)

    labels = {}  # Casting keys from str to int
    for label, name in config['labels'].items():
        labels[int(label)] = name
    config['labels'] = labels

    if 'labels_order' not in config:  # Default order
        config['labels_order'] = list(labels.keys())

    return config


def classify_images(model: torch.nn.Module, dataloader: DataLoader, use_cuda: bool) -> List[Tuple[str, int]]:
    """
    Method run trained model with testing data and returns results.
    :param model: trained model
    :param dataloader: Dataloader with testing data
    :param use_cuda: if true, runs classification with GPU device
    :return: List of tuples with image identifier and predicted label
    """
    result = []
    device = get_device(use_cuda)
    model = model.to(device)
    with torch.no_grad():
        for images, images_ids in tqdm(dataloader, desc='Classifying images', file=sys.stdout):
            images = images.to(device)
            predictions = model(images)
            labels = torch.argmax(predictions, dim=-1)
            for image_id, label in zip(images_ids, labels):
                result.append((image_id, label.item()))

    return result


def save_result(results: List[Tuple[str, int]], config: Dict[str, Any], save_path: str) -> None:
    """
    Method saves classification results into CSV file with labels in one-hot encoding format
    :param results: List with classification results
    :param config: config with labels data
    :param save_path: path where results should be saved
    """
    image_ids, labels = zip(*results)
    result_data = pd.get_dummies(pd.Series(labels))  # Get one-hot encoding
    result_data = result_data[config['labels_order']]  # Change labels order
    result_data = result_data.rename(config['labels'], axis='columns')
    result_data.insert(0, 'file_name', image_ids)
    result_data.to_csv(save_path, index=False)
    print(f'Results saved in {save_path}.')


def main() -> None:
    """
    Method loads trained model and testing data from specified paths, and runs classification.
    Results are then saved into a CSV file.
    """
    args = parse_args()
    config = read_config(args.config_path)

    dataloader = load_test_data(args.images_path, args.batch)
    checkpoint = torch.load(args.model_path)
    model = resnet50(pretrained=False, num_classes=len(config['labels']))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    del checkpoint

    result = classify_images(model, dataloader, use_cuda=args.use_gpu)
    save_result(result, config, args.save_path)


if __name__ == '__main__':
    main()
