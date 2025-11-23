# Quick Start Guide

Get the ASL Sign Language to Text project running in 5 minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB+ recommended)
- Webcam (for real-time detection)
- Internet connection (for dataset download)

## 🚀 Quick Setup (3 Commands)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup & Test
```bash
python setup.py
```

This will:
- Check all dependencies
- Create necessary directories
- Run architecture demo
- Verify GPU support

### 3. Download Dataset (Optional - Required for Training)
```bash
python download_data.py --dataset alphabet
```

## 🎯 Usage Options

### Option A: Train Your Own Model (~1 hour)

```bash
# Download dataset first (if not done)
python download_data.py --dataset alphabet

# Train the model
python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10

# Run application
python app.py
```

### Option B: Use Pre-trained Model (Recommended)

If you have a pre-trained `asl_model.pth`:
```bash
# Place asl_model.pth in project root
python app.py
```

### Option C: Test Architecture Only

```bash
python demo.py
```

## 🖥️ Application Features

Once running, you can:
1. **Start Camera** - Real-time ASL recognition
2. **Load Image** - Analyze a single image
3. **Load Video** - Process video files
4. **Clear Text** - Reset detected text

## 📊 Expected Performance

- **GPU Training**: ~30-45 minutes for 10 epochs
- **CPU Training**: ~2-4 hours for 10 epochs
- **Inference**: 
  - GPU: ~5-10 ms per image
  - CPU: ~50-100 ms per image

## 🔧 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "Kaggle API error"
1. Go to https://www.kaggle.com/account
2. Create New API Token
3. Place `kaggle.json` in `~/.kaggle/` folder

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --data_dir ... --batch_size 16
```

### "Camera not detected"
- Check camera permissions
- Try different camera index in `app.py`: `cv2.VideoCapture(1)`

## 📁 Project Files

```
asl-sign-language/
├── model.py           # CNN architecture
├── train.py           # Training script
├── app.py             # Desktop GUI application
├── demo.py            # Architecture demo
├── download_data.py   # Dataset downloader
├── setup.py           # Setup & verification
├── requirements.txt   # Dependencies
└── README.md          # Full documentation
```

## 🎓 For Students/Portfolio

This project demonstrates:
- ✓ Deep Learning with PyTorch
- ✓ Computer Vision with OpenCV
- ✓ Desktop Application Development
- ✓ Real-time Processing
- ✓ GPU Acceleration
- ✓ Data Pipeline Management

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Model not loading | Train model first or download pre-trained |
| Low accuracy | Train for more epochs (20-30) |
| Slow inference | Enable GPU support |
| Dataset error | Check Kaggle API setup |

## 💡 Pro Tips

1. **GPU Users**: Install CUDA-enabled PyTorch for 10x faster training
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Limited GPU Memory**: Use smaller batch sizes (16 or 8)

3. **Better Accuracy**: Train for 20-30 epochs with data augmentation

4. **Production Use**: Add confidence threshold adjustment in `app.py`

## 📧 Need Help?

Check the full README.md for detailed documentation and troubleshooting.

---

**Ready to go!** Start with `python setup.py` to verify everything is working.