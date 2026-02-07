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
import torch.nn.functional as F


class VariancePenalizedBCELoss(nn.Module):
    """
    Custom loss combining:
    1. Binary Cross-Entropy for safe/unsafe classification.
    2. Variance penalty in flat regions to encourage stable predictions.
    """
    def __init__(self, bce_weight=1.0, variance_weight=0.1):
        super().__init__()
        self.bce_weight = bce_weight
        self.variance_weight = variance_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, labels, depths=None):
        # 1. BCE Loss
        bce_loss = self.bce(outputs, labels)
        
        # 2. Variance Penalty in Flat Regions
        # If depth is provided, use it to identify flat regions
        # Flat = low variance in depth. We penalize high prediction variance in those regions.
        variance_loss = 0.0
        
        if depths is not None:
            probs = torch.sigmoid(outputs)
            
            # Compute local variance of predictions using avg pooling trick
            # Var(X) = E[X^2] - E[X]^2
            kernel_size = 5
            padding = kernel_size // 2
            
            mean_pred = F.avg_pool2d(probs, kernel_size, stride=1, padding=padding)
            mean_sq_pred = F.avg_pool2d(probs ** 2, kernel_size, stride=1, padding=padding)
            pred_variance = mean_sq_pred - mean_pred ** 2
            pred_variance = torch.clamp(pred_variance, min=0)  # Numerical stability
            
            # Compute flatness mask from depth (low depth variance = flat)
            # depths shape: (B, H, W) or (B, 1, H, W)
            if depths.dim() == 3:
                depths = depths.unsqueeze(1)
            
            mean_depth = F.avg_pool2d(depths, kernel_size, stride=1, padding=padding)
            mean_sq_depth = F.avg_pool2d(depths ** 2, kernel_size, stride=1, padding=padding)
            depth_variance = mean_sq_depth - mean_depth ** 2
            depth_variance = torch.clamp(depth_variance, min=0)
            
            # Flat regions = where depth_variance is below threshold
            flat_threshold = 0.01  # Tune as needed
            flat_mask = (depth_variance < flat_threshold).float()
            
            # Penalize high prediction variance in flat regions
            variance_loss = (pred_variance * flat_mask).mean()
        
        total_loss = self.bce_weight * bce_loss + self.variance_weight * variance_loss
        return total_loss

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
    
    # Freeze Backbone (Train Only Decoder)
    model.freeze_backbone()

    # Loss & Optimizer (Custom Variance-Penalized BCE)
    criterion = VariancePenalizedBCELoss(bce_weight=1.0, variance_weight=0.1)
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

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
            
            # Depths: prepare for variance penalty
            depths_tensor = None
            if depths is not None and depths[0] is not None:
                depths_tensor = torch.stack([torch.tensor(d) for d in depths]).float().to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels, depths_tensor)
            
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
                
                depths_tensor = None
                if depths is not None and depths[0] is not None:
                    depths_tensor = torch.stack([torch.tensor(d) for d in depths]).float().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels, depths_tensor)
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
    parser.add_argument("--epochs", type=int, default=15)  # Prototype: 10-20
    parser.add_argument("--batch_size", type=int, default=4)  # Small batch (4-8) for low RAM
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=256)
    
    args = parser.parse_args()
    train(args)
