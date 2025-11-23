# ASL Sign Language to Text Converter

A desktop application that converts American Sign Language (ASL) hand signs to text using deep learning. Built with PyTorch and OpenCV, supporting real-time camera input, image processing, and video analysis.

## Features

- **Real-time Camera Detection**: Live webcam feed with ASL recognition
- **Image Processing**: Upload and analyze static images
- **Video Processing**: Process pre-recorded videos
- **GPU Acceleration**: Utilizes CUDA for faster inference
- **29 Classes**: Recognizes A-Z letters + space, delete, and nothing gestures

## Project Structure

```
asl-sign-language/
├── model.py              # CNN architecture definition
├── train.py              # Model training script
├── app.py                # Desktop application (GUI)
├── download_data.py      # Dataset download utility
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### 1. Clone or Create Project Directory

```bash
mkdir asl-sign-language
cd asl-sign-language
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Setup Kaggle API (for dataset download)

1. Create a Kaggle account at https://www.kaggle.com
2. Go to Account → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/` directory (Linux/Mac) or `C:\Users\<YourUsername>\.kaggle\` (Windows)
4. Set permissions (Linux/Mac only):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Step 1: Download Dataset

```bash
python download_data.py --dataset alphabet
```

This downloads the ASL Alphabet dataset (~1.1GB) with 87,000 training images.

### Step 2: Train the Model

```bash
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10 --batch_size 32
```

Training takes approximately 30-60 minutes with GPU. The trained model is saved as `asl_model.pth`.

**Training Parameters:**
- `--data_dir`: Path to training data
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)

### Step 3: Run the Application

```bash
python app.py
```

## Application Guide

### Main Features

1. **Start Camera**: Begin real-time ASL recognition from webcam
2. **Stop Camera**: Stop the camera feed
3. **Load Image**: Upload a single image for prediction
4. **Load Video**: Process a video file
5. **Clear Text**: Reset the detected text

### Recognition Logic

- The app detects hand signs continuously
- When a prediction is stable for ~0.5 seconds with >70% confidence, it adds the character to text
- **Special characters**:
  - `space`: Adds a space
  - `del`: Deletes the last character
  - `nothing`: No action (idle hand position)

## Model Architecture

**ASLNet CNN Architecture:**
- 3 Convolutional blocks (32, 64, 128 filters)
- Batch normalization after each conv layer
- Max pooling (2x2) after each block
- Dropout (0.25) for regularization
- Fully connected layers: 512 → 29 classes
- Input: 200x200 RGB images
- Output: 29 class probabilities

## Dataset Information

**ASL Alphabet Dataset** (Kaggle):
- 87,000 training images
- 29 classes: A-Z, space, delete, nothing
- 200x200 colored images
- Each class has ~3,000 images

## GPU Support

The application automatically detects and uses CUDA GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

Check GPU availability:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

## Performance Tips

1. **GPU Training**: Use CUDA-enabled GPU for 10-20x faster training
2. **Batch Size**: Increase if you have more GPU memory (try 64 or 128)
3. **Data Augmentation**: Enabled by default (flips, rotations, color jitter)
4. **Confidence Threshold**: Adjust in app.py (default: 70%)

## Troubleshooting

**Model not loading:**
- Ensure `asl_model.pth` exists in the project directory
- Train the model first using `train.py`

**Kaggle API errors:**
- Check `kaggle.json` is in correct location
- Verify Kaggle credentials are valid
- Run `kaggle datasets list` to test connection

**Camera not working:**
- Check camera permissions
- Try changing camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**CUDA out of memory:**
- Reduce batch size
- Use CPU instead: `device = torch.device('cpu')`

## Future Enhancements

- [ ] Add word and sentence recognition
- [ ] Implement MediaPipe hand tracking for better accuracy
- [ ] Support for continuous sign language (not just alphabet)
- [ ] Mobile app version
- [ ] Real-time translation to multiple languages

## Technical Stack

- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision and camera handling
- **Tkinter**: GUI framework
- **Pillow**: Image processing
- **Kaggle API**: Dataset management

## License

Educational project for portfolio purposes.

## Credits

- Dataset: ASL Alphabet by Akash (Kaggle)
- Framework: PyTorch, OpenCV
- Developed: July 2025