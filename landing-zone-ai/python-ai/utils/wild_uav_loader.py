import os
import cv2
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class WildUAVDataset(Dataset):
    """
    WildUAV Dataset Loader
    Structure expected:
    data_root/
        Mapping/
            seq00/
                img/
                    000000.png
                metadata/
                depth/
            ...
    """
    def __init__(self, data_root, transform=None, split='Mapping'):
        self.data_root = data_root
        self.transform = transform
        self.split = split
        self.image_paths = []
        
        search_path = os.path.join(data_root, split, 'seq*', 'img', '*.png')
        self.image_paths = sorted(glob.glob(search_path))
        
        if not self.image_paths:
            # Try searching for JPGs if PNGs not found (Video set)
            search_path_jpg = os.path.join(data_root, split, 'seq*', 'img', '*.jpg')
            self.image_paths = sorted(glob.glob(search_path_jpg))

        print(f"Found {len(self.image_paths)} images in {split} set.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Determine depth path (replace 'img' with 'depth' and extension with .npy)
        depth_path = img_path.replace('img', 'depth').replace('.png', '.npy').replace('.jpg', '.npy')
        
        depth = None
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
        
        if self.transform:
            # Convert to PIL for standard torchvision transforms
            image = Image.fromarray(image)
            image = self.transform(image)
        
        # Load Label if exists
        label_path = img_path.replace('img', 'labels').replace('.jpg', '.png')
        label = None
        if os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            # Normalize to 0-1 tensor? Usually handled by transform or custom logic
            if label is not None:
                label = label / 255.0 # float mask 0.0 - 1.0
            
        return image, depth, label, img_path

if __name__ == "__main__":
    # Test block
    dataset = WildUAVDataset(data_root='../data', split='Mapping')
    if len(dataset) > 0:
        print(f"First image path: {dataset.image_paths[0]}")
