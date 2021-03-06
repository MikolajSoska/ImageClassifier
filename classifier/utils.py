import random

import torch


def set_random_seed(seed: int) -> None:
    """
    Method sets random seed for results reproducibility.
    :param seed: Random seed
    """
    torch.manual_seed(seed)
    random.seed(seed)


def get_device(use_cuda: bool) -> torch.device:
    """
    Method sets torch device (if available) and prints details about it.
    :param use_cuda: If true, method will try to use this GPU device
    :return: Picked device
    """
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_properties = torch.cuda.get_device_properties(device)
            device_name = f'CUDA ({device_properties.name}), ' \
                          f'Total memory: {round(device_properties.total_memory / (1024 ** 2), 2):g} MB'
        else:
            print('CUDA device is not available.')
            device = torch.device('cpu')
            device_name = 'CPU'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    print(f'Using device: {device_name}')
    return device
