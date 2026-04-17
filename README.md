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
├── PROJECT_SUMMARY.md
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

## Usage

### Download Dataset

```bash
python download_data.py --dataset alphabet
```

### Train the Model

```bash
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10 --batch_size 32
```

The trained model is saved as `asl_model.pth`.

### Run the Application

```bash
python app.py
```

## Application Guide

- Start Camera: start live webcam ASL recognition
- Stop Camera: stop the webcam feed
- Load Image: predict an uploaded image
- Load Video: predict signs from a video
- Clear Text: reset detected text output

## Recognition Logic

- The app detects hand signs continuously
- If a prediction remains stable with confidence above 70%, it adds the character to the text
- Special characters:
  - `space`: adds a space
  - `del`: deletes the last character
  - `nothing`: no action

## Model Architecture

- 3 convolutional blocks with 32, 64, and 128 filters
- Batch normalization after each convolution
- Max pooling layers
- Dropout regularization
- Fully connected classifier with 29 output classes
- Input image size: 200x200 RGB

## Dataset Information

The ASL Alphabet dataset from Kaggle includes:

- 87,000 training images
- 29 classes: A-Z, space, delete, nothing
- 200x200 color images

## GPU Support

The application uses CUDA if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## Performance Tips

- Use a GPU-enabled PyTorch build for faster training
- Increase batch size if GPU memory allows
- Use data augmentation for better results
- Adjust the confidence threshold in `app.py` if needed

## Troubleshooting

### Model not found

- Make sure `asl_model.pth` exists
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
