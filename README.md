
# Object Detection Under Adversarial and Noise Corruption

This project studies how object detection changes when images are:

1. attacked with an adversarial patch,
2. corrupted with different noise types,
3. optionally restored with an HGD denoiser.

The goal is to measure how robust detection is under these conditions using IoU-based comparisons, recall, and precision.

## What This Project Does

- Loads a Faster R-CNN detector.
- Generates adversarial examples from Cityscapes images.
- Applies noise (Gaussian, Salt-and-Pepper, Speckle, Poisson).
- Applies denoising pipelines:
	- `noise_only`
	- `denoise_only`
	- `noise_then_denoise`
- Compares detections with IoU metrics (`standard`, `giou`, `diou`, `ciou`).
- Reports recall and precision (with optional IoU margin tolerance).

## Project Structure

- `src/` - core Python code
	- `adversarial_detection.py` - detector wrapper and adversarial patch generation
	- `iou_metrics.py` - IoU calculations and matching logic
	- `hgd_trainer.py` - HGD denoiser training logic
	- `noise_functions.py` - image noise generation
- `notebooks/` - experiments and analysis
	- `hgd_training.ipynb` - train denoiser model
	- `recall_precision.ipynb` - main recall/precision evaluation
	- `batch_test.ipynb` - plotting and batch analysis utilities
- `leftImg8bit/` - dataset images (train/val/test)
- `results/` - generated plots and output artifacts

## Dataset

The dataset is of the cityscapes dataset of leftImg8bit set
This project expects Cityscapes-style image folders in:

- `leftImg8bit/train`
- `leftImg8bit/val`
- `leftImg8bit/test`

## Quick Start

1. Create and activate a virtual environment.
2. Install required packages (PyTorch, torchvision, numpy, matplotlib, pandas, jupyter).
3. Open the notebooks and run cells from top to bottom.

Recommended order:

1. `notebooks/hgd_training.ipynb` (if training/refreshing denoiser)
2. `notebooks/recall_precision.ipynb` (main metrics)
3. `notebooks/batch_test.ipynb` (extra plots/analysis)

## Main Outputs

- Trained denoiser weights:
	- `notebooks/hgd_denoiser_cityscapes_best.pth`
	- `notebooks/hgd_denoiser_cityscapes.pth`
- Evaluation figures and statistics in `results/`

## License

See `license.txt`.
