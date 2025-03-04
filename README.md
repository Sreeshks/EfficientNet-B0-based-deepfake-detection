# EfficientNet-B0 Deepfake Detection System

A state-of-the-art deepfake detection system leveraging EfficientNet-B0 architecture for efficient and accurate identification of AI-manipulated videos.

## Overview

This project implements an advanced deepfake detection system using the EfficientNet-B0 architecture. The system is designed to analyze facial regions in videos and determine whether they contain AI-manipulated content. It offers high accuracy with lower computational requirements compared to heavier models.

## Features

- **Efficient Architecture**: Utilizes EfficientNet-B0 as the backbone, providing excellent performance with lower computational cost
- **Face Detection**: Implements MTCNN (Multi-task Cascaded Convolutional Networks) for precise face detection
- **Frame Sampling**: Intelligent frame sampling strategy to reduce processing time while maintaining accuracy
- **Data Augmentation**: Employs rotation and color jitter techniques to improve model robustness
- **Production-Ready**: Complete pipeline from training to inference with comprehensive evaluation metrics

## Requirements

- Python 3.6+
- PyTorch 1.7.0+
- torchvision
- facenet-pytorch
- OpenCV
- NumPy
- Pillow

## Installation

```bash
git clone https://github.com/yourusername/efficientnet-deepfake-detection.git
cd efficientnet-deepfake-detection
pip install -r requirements.txt
```

## Dataset

This project uses the "1000 Videos Split" dataset available on Kaggle:
[https://www.kaggle.com/datasets/nanduncs/1000-videos-split](https://www.kaggle.com/datasets/nanduncs/1000-videos-split)

The dataset contains a balanced collection of real and fake videos organized into training, validation, and testing splits, making it ideal for developing and evaluating deepfake detection models.

## Dataset Structure

The code expects the dataset to be organized as follows:

```
dataset_root/
├── train/
│   ├── real/
│   │   └── [real video files]
│   └── fake/
│       └── [fake video files]
├── test/
│   ├── real/
│   └── fake/
└── validation/
    ├── real/
    └── fake/
```

## Usage

### Training the Model

```python
from efficientnet_deepfake import DeepfakeDetectionModel, FaceForensicsDataset, train_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Data preprocessing with augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = FaceForensicsDataset("path/to/train_data", transform=transform)
val_dataset = FaceForensicsDataset("path/to/val_data", transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Initialize and train model
model = DeepfakeDetectionModel()
train_model(model, train_loader, val_loader, num_epochs=10, save_path="deepfake_efficientnet.pth")
```

### Testing the Model

```python
from efficientnet_deepfake import DeepfakeDetectionModel, FaceForensicsDataset, test_model
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Create test dataset and loader
test_dataset = FaceForensicsDataset("path/to/test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load model and test
model = DeepfakeDetectionModel()
model.load_state_dict(torch.load("deepfake_efficientnet.pth", 
                     map_location='cuda' if torch.cuda.is_available() else 'cpu'))
test_accuracy = test_model(model, test_loader)
```

### Analyzing a Video

```python
from efficientnet_deepfake import DeepfakeDetectionModel, predict_video
import torchvision.transforms as transforms
import torch

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load model
model = DeepfakeDetectionModel()
model.load_state_dict(torch.load("deepfake_efficientnet.pth", 
                     map_location='cuda' if torch.cuda.is_available() else 'cpu'))

# Analyze video
result = predict_video(model, "path/to/video.mp4", transform, frame_sample_rate=5)
if result:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence score: {result['avg_probability_fake']:.2f}")
```

## Model Architecture

The system uses EfficientNet-B0 with the following modifications:
- Pretrained weights from ImageNet
- Custom classifier head with dropout for regularization
- Two fully connected layers (backbone features → 512 → 2)

EfficientNet-B0 was chosen for its excellent balance between accuracy and computational efficiency, making it suitable for deployment in various environments including those with limited computational resources.

## Performance Factors

The system's performance is influenced by:
- Face detection quality and consistency
- Frame sampling rate
- Video quality and compression artifacts
- Dataset diversity and balance
- Augmentation strategy during training

## Future Enhancements

- Multi-frame temporal analysis
- Ensemble with other model architectures
- Attention mechanisms to focus on manipulation artifacts
- Integration of audio analysis for multi-modal detection
- Explainable AI features to highlight manipulated regions


## Acknowledgements

- The facenet-pytorch library for MTCNN implementation
- PyTorch for the pretrained EfficientNet-B0 model
- Dataset provided by [1000 Videos Split](https://www.kaggle.com/datasets/nanduncs/1000-videos-split) on Kaggle
