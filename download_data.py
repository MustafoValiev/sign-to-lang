import os
import kaggle
import zipfile
import shutil

def download_asl_dataset():
    print("Downloading ASL Alphabet dataset from Kaggle...")
    print("Make sure you have kaggle.json in ~/.kaggle/ directory")
    
    dataset_name = "grassknoted/asl-alphabet"
    download_path = "./data"
    
    os.makedirs(download_path, exist_ok=True)
    
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"Dataset downloaded and extracted to {download_path}")
        
        train_path = os.path.join(download_path, "asl_alphabet_train", "asl_alphabet_train")
        if os.path.exists(train_path):
            print(f"\nTraining data found at: {train_path}")
            classes = os.listdir(train_path)
            print(f"Number of classes: {len(classes)}")
            print(f"Classes: {sorted(classes)}")
            
            total_images = sum([len(os.listdir(os.path.join(train_path, c))) 
                              for c in classes if os.path.isdir(os.path.join(train_path, c))])
            print(f"Total training images: {total_images}")
        
        return train_path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTo use Kaggle API:")
        print("1. Create a Kaggle account at https://www.kaggle.com")
        print("2. Go to Account settings and create an API token")
        print("3. Place kaggle.json in ~/.kaggle/ directory")
        print("4. chmod 600 ~/.kaggle/kaggle.json")
        return None

def download_sign_language_mnist():
    print("\nAlternative: Downloading Sign Language MNIST (smaller dataset)...")
    
    dataset_name = "datamunge/sign-language-mnist"
    download_path = "./data_mnist"
    
    os.makedirs(download_path, exist_ok=True)
    
    try:
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print(f"MNIST dataset downloaded to {download_path}")
        return download_path
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download ASL datasets from Kaggle')
    parser.add_argument('--dataset', type=str, default='alphabet', 
                       choices=['alphabet', 'mnist'], 
                       help='Which dataset to download')
    args = parser.parse_args()
    
    if args.dataset == 'alphabet':
        download_asl_dataset()
    else:
        download_sign_language_mnist()