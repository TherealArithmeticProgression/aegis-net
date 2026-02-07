import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from utils.wild_uav_loader import WildUAVDataset

def process_dataset(dataset, output_dir, target_size=(256, 256)):
    """
    Resizes images and depth maps to target_size to ensure alignment.
    Saves RGB as .png and Depth as .npy.
    """
    # Create output directories
    img_out_dir = os.path.join(output_dir, 'img')
    depth_out_dir = os.path.join(output_dir, 'depth')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(depth_out_dir, exist_ok=True)

    print(f"Processing {len(dataset)} samples...")
    print(f"Target Size: {target_size}")
    
    for i in tqdm(range(len(dataset))):
        img_path = dataset.image_paths[i]
        
        # 1. Load and Resize RGB Image
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        # Resize RGB (Inter_Area is good for shrinking)
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # 2. Load and Resize Depth Map (if exists)
        depth_path = img_path.replace('img', 'depth').replace('.png', '.npy').replace('.jpg', '.npy')
        
        if os.path.exists(depth_path):
            depth = np.load(depth_path)
            # Resize Depth (Nearest to preserve specific values if categorical, or Linear/Area if continuous)
            # Assuming continuous depth for UAV landing zones
            resized_depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Save Depth
            # Maintain directory structure relative to 'depth' folder or flat structure?
            # Let's use a flat structure with unique names or replicate original tree.
            # For simplicity here, we replicate relative path structure.
            rel_path = os.path.relpath(img_path, dataset.data_root) # e.g. Mapping/seq00/img/000000.png
            
            # Construct save paths
            # We want: output/Mapping/seq00/img/000000.png
            # And:     output/Mapping/seq00/depth/000000.npy
            
            save_rel_base = os.path.splitext(rel_path)[0] # Mapping/seq00/img/000000
            save_rel_base = save_rel_base.replace('img', '') # Mapping/seq00//000000 (roughly)
            
            # cleaner way: keep original structure
            save_path_img = os.path.join(output_dir, rel_path)
            save_path_depth = os.path.join(output_dir, rel_path.replace('img', 'depth').replace('.png', '.npy').replace('.jpg', '.npy'))
            
            os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
            os.makedirs(os.path.dirname(save_path_depth), exist_ok=True)
            
            cv2.imwrite(save_path_img, resized_image)
            np.save(save_path_depth, resized_depth)
        else:
            # Save just image if no depth
            save_path_img = os.path.join(output_dir, os.path.relpath(img_path, dataset.data_root))
            os.makedirs(os.path.dirname(save_path_img), exist_ok=True)
            cv2.imwrite(save_path_img, resized_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WildUAV Dataset: Resize & Align")
    parser.add_argument("--root", type=str, default="../data/WildUAV", help="Path to raw WildUAV dataset")
    parser.add_argument("--output", type=str, default="../data/WildUAV_Processed", help="Path to save processed data")
    parser.add_argument("--width", type=int, default=256, help="Target width")
    parser.add_argument("--height", type=int, default=256, help="Target height")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root):
        print(f"Error: Data root '{args.root}' not found.")
        exit(1)

    print(f"Scanning dataset at {args.root}...")
    # Load dataset wrapper just to get paths
    dataset = WildUAVDataset(data_root=args.root, split="Mapping")
    
    process_dataset(dataset, args.output, target_size=(args.width, args.height))
