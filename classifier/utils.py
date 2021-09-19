import random

import torch


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def convert_bytes_to_megabytes(bytes_number: int) -> float:
    return round(bytes_number / (1024 ** 2), 2)


def get_device(use_cuda: bool) -> torch.device:
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_properties = torch.cuda.get_device_properties(device)
            device_name = f'CUDA ({device_properties.name}), ' \
                          f'Total memory: {convert_bytes_to_megabytes(device_properties.total_memory):g} MB'
        else:
            print('CUDA device is not available.')
            device = torch.device('cpu')
            device_name = 'CPU'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'

    print(f'Using device: {device_name}')
    return device