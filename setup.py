import os
import sys
import subprocess

def check_python_version():
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_pytorch():
    print("\nChecking PyTorch installation...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, will use CPU")
            print("  For GPU support, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def check_opencv():
    print("\nChecking OpenCV...")
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not found")
        return False

def create_directories():
    print("\nCreating project directories...")
    dirs = ['data', 'models', 'outputs']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created {d}/")
    return True

def check_kaggle_setup():
    print("\nChecking Kaggle API setup...")
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    
    if os.path.exists(kaggle_path):
        print(f"✓ Kaggle API key found at {kaggle_path}")
        try:
            import kaggle
            print("✓ Kaggle API is working")
            return True
        except Exception as e:
            print(f"⚠ Kaggle API error: {e}")
            return False
    else:
        print("⚠ Kaggle API key not found")
        print("  To download datasets:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Create New API Token")
        print(f"  3. Place kaggle.json in {kaggle_path}")
        return False

def run_demo():
    print("\nRunning architecture demo...")
    try:
        subprocess.check_call([sys.executable, "demo.py"])
        return True
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    print("=" * 60)
    print("ASL Sign Language to Text - Setup Script")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("PyTorch", check_pytorch),
        ("OpenCV", check_opencv),
        ("Directories", create_directories),
        ("Kaggle API", check_kaggle_setup),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("Setup Summary")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! Running demo...")
        run_demo()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Download dataset: python download_data.py --dataset alphabet")
        print("2. Train model: python train.py --data_dir ./data/asl_alphabet_train/asl_alphabet_train --epochs 10")
        print("3. Run application: python app.py")
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("You can still proceed, but some features may not work.")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()