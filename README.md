# Image Segmentation using U-Net

This repository contains a TensorFlow implementation of the U-Net architecture for semantic segmentation tasks. The code is designed to train a U-Net model on image datasets with corresponding segmentation masks.

## Features

- **Data Preprocessing**: Handles loading, resizing, and normalizing images and masks.
- **Model Architecture**: Implements the U-Net model with customizable parameters such as input size and number of classes.
- **Training Pipeline**: Includes training with configurable epochs, batch size, and other hyperparameters.
- **Visualization**: Provides functions for visualizing input images, true masks, and predicted masks.

## Prerequisites

- Python 3.7+
- TensorFlow 2.0+
- Required Python libraries: `numpy`, `pandas`, `imageio`, `matplotlib`

Install the dependencies using:

```bash
pip install tensorflow numpy pandas imageio matplotlib
```

## Usage

1. **Dataset Preparation**:
   - Place input images in the `./data/CameraRGB/` directory.
   - Place corresponding masks in the `./data/CameraMask/` directory.

2. **Run the Script**:
   - Execute the `Image_segmentation_Unet_v2.py` script to train the model and visualize results.
   - The script preprocesses the dataset, trains the U-Net model, and displays predictions.

3. **Model Configuration**:
   - Modify the following parameters in the script if necessary:
     - `img_height`, `img_width`: Input image dimensions (default: 96x128).
     - `num_channels`: Number of channels in input images (default: 3).
     - `EPOCHS`, `BATCH_SIZE`: Training hyperparameters.

4. **Visualizations**:
   - The script generates side-by-side visualizations of input images, true masks, and predicted masks during training.

## Model Summary

The U-Net architecture consists of:
- **Contracting Path**: Downsampling with convolutional layers and max-pooling.
- **Expanding Path**: Upsampling with transposed convolutions and skip connections from the contracting path.
