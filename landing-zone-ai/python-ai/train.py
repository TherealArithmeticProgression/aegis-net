import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.unet_resnet import ResNetUNet
from utils.wild_uav_loader import WildUAVDataset

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Loader
    dataset = WildUAVDataset(data_root=args.data_root, split="Mapping", transform=transform)
    
    # Simple split (80/20) - simplified for demo
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for Windows safety
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = ResNetUNet(n_class=1).to(device)

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, depths, labels, paths in pbar:
            if labels is None:
                continue # Skip if no labels found
                
            images = images.to(device)
            # Labels: Load as float, shape (B, H, W). Add channel dim -> (B, 1, H, W)
            labels = torch.stack([torch.tensor(l) for l in labels]) if isinstance(labels, list) else labels
            labels = labels.unsqueeze(1).float().to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Validation (Simple Loss Check)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, depths, labels, paths in val_loader:
                if labels is None: continue
                images = images.to(device)
                labels = labels.unsqueeze(1).float().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save Checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'landing_model.pth'))

    print("Training Complete. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/WildUAV_Processed", help="Path to processed data")
    parser.add_argument("--save_dir", type=str, default="models", help="Dir to save model weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4) # Small batch for CPU/Consumer GPU
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    
    args = parser.parse_args()
    train(args)
