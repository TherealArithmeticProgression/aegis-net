import os
import cv2
import glob
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
import torch


class UAVAugmentation:
    """
    Data augmentation for UAV aerial images.
    Handles rotations, gaussian noise, and brightness/lighting shifts.
    """
    def __init__(self, 
                 rotation_range=30,      # degrees
                 brightness_range=0.3,   # +/- 30%
                 noise_std=0.05,         # gaussian noise std
                 p_rotate=0.5,
                 p_brightness=0.5,
                 p_noise=0.3):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.noise_std = noise_std
        self.p_rotate = p_rotate
        self.p_brightness = p_brightness
        self.p_noise = p_noise
    
    def __call__(self, image, label=None, depth=None):
        """
        Apply augmentations to image and corresponding label/depth.
        
        Args:
            image: numpy array (H, W, 3)
            label: numpy array (H, W) or None
            depth: numpy array (H, W) or None
        
        Returns:
            Augmented image, label, depth
        """
        # Random Rotation
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image, label, depth = self._rotate(image, label, depth, angle)
        
        # Random Brightness/Lighting Shift
        if random.random() < self.p_brightness:
            factor = 1.0 + random.uniform(-self.brightness_range, self.brightness_range)
            image = self._adjust_brightness(image, factor)
        
        # Gaussian Noise
        if random.random() < self.p_noise:
            image = self._add_gaussian_noise(image)
        
        return image, label, depth
    
    def _rotate(self, image, label, depth, angle):
        """Rotate image, label, and depth by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if label is not None:
            label = cv2.warpAffine(label, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        if depth is not None:
            depth = cv2.warpAffine(depth, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return image, label, depth
    
    def _adjust_brightness(self, image, factor):
        """Adjust brightness by factor (simulates lighting changes during UAV flight)."""
        image = image.astype(np.float32)
        image = image * factor
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    def _add_gaussian_noise(self, image):
        """Add Gaussian noise to simulate sensor noise."""
        image = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, self.noise_std, image.shape)
        image = image + noise
        image = np.clip(image, 0, 1) * 255
        return image.astype(np.uint8)


def apply_histogram_equalization(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to RGB image.
    Better than standard histogram equalization for preserving local contrast.
    
    Args:
        image: numpy array (H, W, 3) in RGB format
    
    Returns:
        Equalized image (H, W, 3)
    """
    # Convert to LAB color space (equalize only L channel)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to RGB
    equalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return equalized


def normalize_depth(depth, method='minmax'):
    """
    Normalize depth map to [0, 1] range.
    
    Args:
        depth: numpy array (H, W)
        method: 'minmax' or 'zscore'
    
    Returns:
        Normalized depth (H, W) as float32
    """
    if depth is None:
        return None
    
    depth = depth.astype(np.float32)
    
    if method == 'minmax':
        # Min-Max normalization to [0, 1]
        d_min = np.min(depth)
        d_max = np.max(depth)
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(depth)
        std = np.std(depth)
        if std > 0:
            depth = (depth - mean) / std
        else:
            depth = depth - mean
    
    return depth


def generate_superpixels(image, n_segments=200, compactness=10):
    """
    Generate SLIC superpixels from image.
    
    Args:
        image: numpy array (H, W, 3) RGB
        n_segments: Approximate number of superpixels
        compactness: Balance between color and spatial proximity (higher = more compact)
    
    Returns:
        segments: (H, W) array of superpixel labels
    """
    from skimage.segmentation import slic
    
    segments = slic(image, n_segments=n_segments, compactness=compactness, 
                    start_label=0, channel_axis=2)
    return segments


def label_to_superpixel_mask(label, segments, threshold=0.5):
    """
    Convert pixel-level binary label to superpixel-based mask.
    Each superpixel takes the majority vote of its constituent pixels.
    This reduces label noise and creates coherent safety zones.
    
    Args:
        label: (H, W) binary label array (0-1)
        segments: (H, W) superpixel segmentation
        threshold: Fraction of safe pixels needed for superpixel to be safe
    
    Returns:
        superpixel_label: (H, W) smoothed label based on superpixels
    """
    if label is None:
        return None
    
    superpixel_label = np.zeros_like(label)
    unique_segments = np.unique(segments)
    
    for seg_id in unique_segments:
        mask = segments == seg_id
        # Calculate fraction of safe pixels in this superpixel
        safe_fraction = np.mean(label[mask])
        # Assign based on majority vote
        superpixel_label[mask] = 1.0 if safe_fraction >= threshold else 0.0
    
    return superpixel_label


def smooth_label_with_superpixels(image, label, n_segments=200, threshold=0.5):
    """
    Full pipeline: Generate superpixels and smooth label.
    
    Args:
        image: RGB image (H, W, 3)
        label: Binary label (H, W)
        n_segments: Number of superpixels
        threshold: Majority vote threshold
    
    Returns:
        Smoothed label (H, W)
    """
    if label is None:
        return None
    
    segments = generate_superpixels(image, n_segments=n_segments)
    smoothed = label_to_superpixel_mask(label, segments, threshold=threshold)
    return smoothed


class WildUAVDataset(Dataset):
    """
    WildUAV Dataset Loader with UAV-specific augmentations and superpixel smoothing.
    """
    def __init__(self, data_root, transform=None, split='Mapping', 
                 augment=True, use_superpixels=True, n_superpixels=200):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.image_paths = []
        
        # Initialize augmentation
        self.augment = augment
        self.augmentation = UAVAugmentation() if augment else None
        
        # Superpixel smoothing for labels
        self.use_superpixels = use_superpixels
        self.n_superpixels = n_superpixels
        
        search_path = os.path.join(data_root, split, 'seq*', 'img', '*.png')
        self.image_paths = sorted(glob.glob(search_path))
        
        if not self.image_paths:
            search_path_jpg = os.path.join(data_root, split, 'seq*', 'img', '*.jpg')
            self.image_paths = sorted(glob.glob(search_path_jpg))

        print(f"Found {len(self.image_paths)} images. Augment: {augment}, Superpixels: {use_superpixels}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth
        depth_path = img_path.replace('img', 'depth').replace('.png', '.npy').replace('.jpg', '.npy')
        depth = None
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
        
        # Load label
        label_path = img_path.replace('img', 'labels').replace('.jpg', '.png')
        label = None
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is not None:
                label = label / 255.0
        
        # === PREPROCESSING ===
        # 1. Apply Histogram Equalization to RGB
        image = apply_histogram_equalization(image)
        
        # 2. Normalize Depth to [0, 1]
        if depth is not None:
            depth = normalize_depth(depth, method='minmax')
        
        # 3. Apply Superpixel Smoothing to Labels
        if self.use_superpixels and label is not None:
            label = smooth_label_with_superpixels(image, label, n_segments=self.n_superpixels)
        
        # === AUGMENTATION ===
        if self.augmentation is not None:
            image, label, depth = self.augmentation(image, label, depth)
        
        # Apply torchvision transforms
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)
            
        return image, depth, label, img_path


if __name__ == "__main__":
    # Test augmentation
    dataset = WildUAVDataset(data_root='../data', split='Mapping', augment=True)
    if len(dataset) > 0:
        print(f"First image path: {dataset.image_paths[0]}")
        img, depth, label, path = dataset[0]
        print(f"Image shape: {img.shape if hasattr(img, 'shape') else 'tensor'}")

