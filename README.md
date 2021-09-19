# ImageClassifier

Simple deep learning model for images classification. Script trains ResNet50 with images from specified path. Training
is performed with cross-entropy as loss function and Adam as optimizer.

Images need to be stored in format:

```
- root_dir:
  - class_1: a.jpg, b.jpg
  - class_2: c.jpg, d.jpg
    ...
```

Trained model is then saved in a checkpoint file and can be used in classification.

## Installation

Code was writter with Python 3.7. Required packages are specified in [`requirements.txt`](requirements.txt) file and can
be installed with:

- `pip install -r requirements.txt`

**Note:** Installation of PyTorch and torchvision packages can differ in various OS and environments, so the best way is
to follow instruction from https://pytorch.org/get-started/locally/.

## Running training

To run training script run:

- `python -m classifier.run_train`

with following arguments:

- `--images-path`: path to training images
- `--epochs`: number of epochs for model training
- `--batch`: batch size during training
- `--val-ratio`: validation ratio for train/val split
- `--lr`: learning rate
- `--adam-betas`: betas values for Adam optimizer
- `--adam-eps`: epsilon values for Adam optimizer
- `--use-gpu`: train with CUDA
- `--logs-path`: directory where Tensorboard logs will be stored
- `--save-path`: directory where saved model will be stored
- `--run-id`: unique identifier for Tensorboard and checkpoint names
- `--seed`: random seed

Details about arguments format and default values is specified in [`classifier/run_train.py`](classifier/run_train.py)
file.

## Running classification

Trained model can be evaluated with testing data. Testing data need to be stored in `.jpg` files in a single directory.
Predictor script is launched with:

- `python -m classifier.predict`

with following arguments:

- `--images-path`: path to testing images
- `--batch`: batch size during testing
- `--use-gpu`: evaluate with CUDA
- `--model-path`: path to saved model file
- `--config-path`: path to JSON config file with label data (details [here](#config-file))
- `--save-path`: path to output CSV file where result will be saved

Details about arguments format and default values is specified in [`classifier/predict.py`](classifier/predict.py)
file.

Results are saved in CSV file in one-hot encoding, e.g.:

```
file_name | class_1 | class_2 | class_3
---------------------------------------
abc.jpg   |    0    |    1    |   0
---------------------------------------
cde.jpg   |    1    |    0    |   0
---------------------------------------
fgh.jpg   |    0    |    1    |   0
```

### Config file

This JSON file tells predictor which label numbers are associated with which class names. Optionally, custom class order
can be specified in this file, which tells predictor in which order columns should be organised in result file.

Config file structure:

```
{
  "labels": {
    "0": "label_1",
    "1": "label_2",
    "2": "label_3"
  },
  "labels_order": [1, 0, 2]
}
```
