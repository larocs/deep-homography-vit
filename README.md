# Deep Homography ViT

This repository contains an implementation of a deep homography estimation model using a Vision Transformer (ViT) backbone, specifically leveraging the TimeSformer architecture for processing image pairs.

## Project Structure

* **`model.py`**: Defines the `HomographyRegressor` class, which uses a `VisionTransformer` to predict the 8 values representing the corner deltas between two image patches.
* **`dataloader.py`**: Implements the `HomographyDataset` class for generating synthetic homography data. It extracts patches from images, applies random perturbations to corners, and warps images to create pairs for training.
* **`train.py`**: The main training script. It handles the training loop, validation, logging via TensorBoard, and saving the best performing model.
* **`test.py`**: A script to evaluate the trained model against traditional feature-based methods like ORB.
* **`utils.py`**: Contains helper functions for coordinate transformations (four-point to homography and vice versa), custom data collation, and visualization of homography estimations.
* **`configs/example_config.yaml`**: A YAML example configuration file for setting hyperparameters such as learning rate, batch size, and ViT architecture details.

## Requirements

The project uses the following main dependencies:

* PyTorch
* OpenCV
* OmegaConf
* TimeSformer (included as a submodule)
* Matplotlib
* Tqdm

## Configuration

Training and model parameters are managed via YAML configuration files. Key parameters include:

* **Data**: `batch_size`, `patch_size`, and `rho` (maximum perturbation).
* **ViT**: `dim_emb`, `depth`, `heads`, and whether to use `pretrained` weights.

## Usage

### Training

To start training, use the `train.py` script and provide a configuration file:

```bash
python train.py --conf configs/example_config.yaml --use_cuda

```

The script will log progress to TensorBoard and save the best model as `best_model.pth`.

### Testing and Evaluation

To evaluate a trained model and compare it with ORB-based estimation:

```bash
python test.py

```

This script generates visualization images in an `output/` directory, showing the base image, the warped image with predicted corners, and the individual patches.

## Model Architecture

The `HomographyRegressor` wraps a Vision Transformer with a regression head:

1. It takes two RGB images (view0 and view1).
2. The images are stacked and passed through the ViT backbone.
3. A global average pooling is applied to the ViT tokens.
4. A linear regressor outputs 8 values corresponding to the  displacements of the four patch corners.

## Model Weights
The weights can be downloaded [here](https://huggingface.co/hudsonmsb/deep_homography_vit/resolve/main/deep_homography_vit.pth).

