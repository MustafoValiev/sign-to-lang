# ASL Sign Language to Text Converter

A desktop application that converts American Sign Language (ASL) hand signs to text using deep learning. It is built with PyTorch, OpenCV, and Tkinter, and supports webcam input, image prediction, and video analysis.

## Features

- Real-time camera detection for ASL gestures
- Image upload and prediction
- Video file prediction
- GPU support with CUDA when available
- Recognizes 29 classes: A-Z, space, delete, and nothing

## Project Structure

```
sign-lang-to-text/
├── app.py
├── create_test_data.py
├── demo.py
├── download_data.py
├── model.py
├── README.md
├── requirements.txt
├── setup.py
└── train.py
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install GPU-enabled PyTorch (optional)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Setup Kaggle API

1. Create a Kaggle account at https://www.kaggle.com
2. Create a new API token
3. Place `kaggle.json` in `~/.kaggle/` on Linux/Mac or `C:\Users\<YourUsername>\.kaggle\` on Windows
4. Set permissions on Linux/Mac:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## Workflow

### Data Preparation

Option A: Download the ASL Alphabet dataset from Kaggle:

```bash
python download_data.py --dataset alphabet
```

Option B: Create synthetic test data for quick testing:

```bash
python create_test_data.py --samples 100
```

### Train the Model

```bash
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10 --batch_size 32 --lr 0.001
```

For CPU-only training, reduce the batch size:

```bash
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10 --batch_size 8
```

### Run the Application

```bash
python app.py
```

## Model Architecture

The project uses a custom CNN called ASLNet with the following structure:

- Input: 3 x 200 x 200 RGB image
- Conv Block 1: 32 filters, batch norm, ReLU, max pool, dropout(0.25)
- Conv Block 2: 64 filters, batch norm, ReLU, max pool, dropout(0.25)
- Conv Block 3: 128 filters, batch norm, ReLU, max pool, dropout(0.25)
- Fully connected layer: 80,000 -> 512, batch norm, ReLU, dropout(0.5)
- Output layer: 512 -> 29 classes

## Training Pipeline

- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: ReduceLROnPlateau
- Data augmentation: random rotation, color jitter, normalization

## Application Guide

- Start Camera: start live webcam ASL recognition
- Stop Camera: stop the webcam feed
- Load Image: predict an uploaded image
- Load Video: predict signs from a video
- Clear Text: reset detected text output

## Recognition Logic

- The app detects hand signs continuously
- If a prediction remains stable for 15 frames and confidence is above 70%, the character is added to text
- Special predictions:
  - `space`: adds a space
  - `del`: deletes the last character
  - `nothing`: no action

## Performance Benchmarks

### Training (10 epochs, 87K images)

- GPU training: 30-45 minutes with a high-end GPU
- CPU training: 2-4 hours with a mid-range CPU

### Inference

- GPU inference: about 5-10 ms per image
- CPU inference: about 50-100 ms per image

## Configuration Options

### Training parameters

- `--data_dir`: Path to training data
- `--epochs`: Number of epochs
- `--batch_size`: Batch size
- `--lr`: Learning rate

### Application settings

In `app.py`, adjust:

- camera index
- confidence threshold
- stability frame count

## GPU Support

The application uses CUDA if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Troubleshooting

### Model not found

- Ensure `asl_model.pth` exists
- Train the model first using `train.py`

### Kaggle API errors

- Verify that `kaggle.json` is in the correct location
- Confirm Kaggle credentials are valid
- Run `kaggle datasets list` to test the connection

### Camera not working

- Check camera permissions
- Try a different camera index in `app.py`: `cv2.VideoCapture(1)`

### CUDA out of memory

- Reduce `batch_size`
- Use CPU only with `device = torch.device('cpu')`

## Future Work

- Add word and sentence recognition
- Implement MediaPipe hand tracking
- Support continuous sign language recognition
- Add a mobile version
- Add real-time translation to other languages

## Technical Stack

- PyTorch
- OpenCV
- Tkinter
- Pillow
- Kaggle API

## License

This project is for educational use.

## Credits

- ASL Alphabet dataset from Kaggle
- Developed using PyTorch and OpenCV
