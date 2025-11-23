import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
from model import ASLNet 

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        
        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, img_name), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_device():
    """Helper to get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Success: NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Success: Apple Silicon GPU (MPS) detected.")
    else:
        device = torch.device('cpu')
        print("⚠️ WARNING: No GPU detected. Training will be slow on CPU.")
    return device

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001):
    device = get_device()
    print(f'Active processing device: {device}')
    
    transform_train = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ASLDataset(data_dir, transform=transform_train)
    
    num_workers = 4 if device.type == 'cuda' else 2
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True if device.type == 'cuda' else False
    )
    
    model = ASLNet(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    print(f'Classes: {train_dataset.classes}')
    print(f'Total samples: {len(train_dataset)}')
    print("Starting training loop...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        scheduler.step(epoch_loss)
        
        print(f'Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': train_dataset.classes,
        'class_to_idx': train_dataset.class_to_idx
    }, 'asl_model.pth')
    
    print('Training completed! Model saved as asl_model.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)