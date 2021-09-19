import argparse

import torch
import torch.nn as nn
from torchvision.models import resnet50

from classifier.data import load_data
from classifier.trainer import Trainer
from classifier.utils import set_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parameters for training image classifier.')
    parser.add_argument('--images-path', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='How much training data use for validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--adam-betas', type=float, default=(0.9, 0.999), nargs=2, help='Betas for Adam optimizer')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon value for Adam optimizer')
    parser.add_argument('--use-gpu', action='store_true', help='Train with CUDA')
    parser.add_argument('--logs-path', type=str, default='./logs', help='Directory for Tensorboard logs')
    parser.add_argument('--save-path', type=str, default='./model', help='Directory for model checkpoints')
    parser.add_argument('--run-id', type=str, default='classifier', help='ID for run identification')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    train_loader, validation_loader, classes_number = load_data(args.images_path, args.batch, args.val_ratio)
    model = resnet50(pretrained=False, num_classes=classes_number)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.adam_betas, eps=args.adam_eps)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_data=train_loader,
        val_data=validation_loader,
        epochs=args.epochs,
        use_cuda=args.use_gpu,
        log_directory=args.logs_path,
        save_directory=args.save_path,
        run_id=args.run_id
    )
    trainer.train()


if __name__ == '__main__':
    main()
