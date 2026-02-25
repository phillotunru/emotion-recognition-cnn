# Emotion Recognition CNN (PyTorch)

A convolutional neural network (CNN) for facial emotion recognition using grayscale face images. This project trains a 7-class emotion classifier in PyTorch with a train/validation pipeline and checkpoint saving/resume support.

## Overview

This model predicts facial emotions from images by learning local visual features (eyes, mouth, facial contours) with convolutional layers. The training setup uses grayscale face images resized to **48x48**, and the final model is a compact CNN designed to reduce overfitting while keeping good accuracy. 

## Model Architecture

The CNN uses:

* **3 convolutional blocks**

  * Conv2D + BatchNorm + ReLU + MaxPool
  * Filters: **32 → 64 → 128**
* **Dropout (0.4)** before/around the fully connected layer to reduce overfitting
* Fully connected layer with **256 units**
* Final output layer with **7 classes** (emotion labels)

This final architecture was chosen after testing a deeper model that overfit too much. 

## Training Setup

* **Framework:** PyTorch
* **Input preprocessing:** resize to 48x48, convert to grayscale, convert to tensor
* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate = **0.001**)
* **Epochs:** 40 (final baseline training run)

The training script also supports:

* checkpoint saving
* resuming training from a saved checkpoint
* configurable train/validation paths via command-line arguments

## Results (Baseline CNN)

After training for 40 epochs, the baseline CNN achieved approximately:

* **Train Accuracy:** ~73%
* **Validation Accuracy:** ~70%

Training and validation loss both decreased consistently, with a small train/val gap indicating mild overfitting but reasonable generalization. 


## Dataset Location

The dataset is not included in this repository and is stored separately (Google Drive).

After downloading the dataset, organize it like this (example):
```text
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── validation/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```
## How to Run

Example:

```bash
python train_emotion_cnn.py \
  --train_dir "path/to/train" \
  --val_dir "path/to/validation" \
  --save_path "model_checkpoint.pth"
```

## Repository Contents

* `train_emotion_cnn.py` — training/validation script for the CNN
* `README.md` — project documentation


