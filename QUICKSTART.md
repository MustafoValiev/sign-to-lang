# Quick Start Guide

This guide helps you run the ASL Sign Language to Text project.

## Prerequisites

- Python 3.8 or higher
- Webcam for real-time detection
- Internet connection for downloading the dataset

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Environment

```bash
python setup.py
```

### 3. Download Dataset

```bash
python download_data.py --dataset alphabet
```

## Usage Options

### Option A: Train the Model

```bash
python download_data.py --dataset alphabet
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10
python app.py
```

### Option B: Use a Pre-trained Model

If `asl_model.pth` is available in the project root:

```bash
python app.py
```

### Option C: Run the Demo

```bash
python demo.py
```

## Application Features

- Start Camera for live ASL recognition
- Load Image for single-image prediction
- Load Video for video file prediction
- Clear Text to reset the output

## Expected Performance

- GPU training: about 30-45 minutes for 10 epochs
- CPU training: about 2-4 hours for 10 epochs

## Troubleshooting

### Missing PyTorch

```bash
pip install torch torchvision
```

### Kaggle API Setup

1. Create a Kaggle account
2. Create an API token
3. Place `kaggle.json` in `~/.kaggle/`

### CUDA Out of Memory

Reduce the batch size:

```bash
python train.py --data_dir ... --batch_size 16
```

### Camera Not Detected

- Check camera permissions
- Try a different camera index in `app.py`: `cv2.VideoCapture(1)`

## Project Files

```
sign-lang-to-text/
├── app.py
├── create_test_data.py
├── demo.py
├── download_data.py
├── model.py
├── PROJECT_SUMMARY.md
├── QUICKSTART.md
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Notes

Check `README.md` for full project documentation and troubleshooting.
