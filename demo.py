import torch
import torch.nn as nn
from model import ASLNet
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

def test_model_architecture():
    print("=" * 50)
    print("ASL Sign Language to Text - Architecture Demo")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print("\n" + "=" * 50)
    print("Model Architecture")
    print("=" * 50)
    
    model = ASLNet(num_classes=29)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model created successfully")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\nModel Architecture:")
    print(model)
    
    print("\n" + "=" * 50)
    print("Testing Forward Pass")
    print("=" * 50)
    
    dummy_input = torch.randn(1, 3, 200, 200).to(device)
    print(f"\n✓ Input shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Number of classes: {output.shape[1]}")
    
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)
    
    print(f"✓ Predicted class: {predicted_class.item()}")
    print(f"✓ Confidence: {confidence.item() * 100:.2f}%")
    
    print("\n" + "=" * 50)
    print("Testing Batch Processing")
    print("=" * 50)
    
    batch_size = 8
    batch_input = torch.randn(batch_size, 3, 200, 200).to(device)
    
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        batch_output = model(batch_input)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            print(f"✓ Batch inference time: {inference_time:.2f} ms")
            print(f"✓ Per-image inference: {inference_time/batch_size:.2f} ms")
    
    print(f"✓ Batch output shape: {batch_output.shape}")
    
    print("\n" + "=" * 50)
    print("Camera Test")
    print("=" * 50)
    
    print("\nAttempting to access camera...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✓ Camera detected and opened successfully")
        ret, frame = cap.read()
        if ret:
            print(f"✓ Frame captured: {frame.shape}")
            print("✓ Camera is working properly")
        cap.release()
    else:
        print("✗ No camera detected (this is okay for server environments)")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print("\n✓ Model architecture: Working")
    print("✓ GPU support:", "Enabled" if torch.cuda.is_available() else "CPU only")
    print("✓ Forward pass: Working")
    print("✓ Batch processing: Working")
    print("\nThe model is ready for training!")
    print("\nNext steps:")
    print("1. Run: python download_data.py --dataset alphabet")
    print("2. Run: python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train")
    print("3. Run: python app.py")
    print("\n" + "=" * 50)

def create_sample_prediction():
    print("\n" + "=" * 50)
    print("Sample Prediction Demo")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLNet(num_classes=29).to(device)
    model.eval()
    
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']
    
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dummy_image = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
    
    with torch.no_grad():
        image_tensor = transform(dummy_image).unsqueeze(0).to(device)
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        
        top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        print("\nTop 5 predictions (random input):")
        for i in range(5):
            print(f"{i+1}. {classes[top5_idx[0][i]]}: {top5_prob[0][i].item()*100:.2f}%")

if __name__ == '__main__':
    test_model_architecture()
    create_sample_prediction()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)