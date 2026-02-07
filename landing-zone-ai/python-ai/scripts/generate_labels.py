import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import glob

def compute_roughness(depth_map, kernel_size=15):
    """
    Computes local roughness (standard deviation) of the depth map.
    Input: Depth map (H, W) or (H, W, 1)
    Output: Roughness map (H, W)
    """
    if len(depth_map.shape) == 3:
        depth_map = depth_map[:, :, 0]
        
    # Standard deviation filter
    # E[X^2] - (E[X])^2
    mean = cv2.blur(depth_map, (kernel_size, kernel_size))
    mean_sq = cv2.blur(depth_map**2, (kernel_size, kernel_size))
    variance = mean_sq - mean**2
    variance[variance < 0] = 0 # Numerical stability
    std_dev = np.sqrt(variance)
    
    return std_dev

def generate_labels(input_dir, output_dir, rough_threshold=5.0):
    """
    Generates binary safety masks based on depth roughness.
    """
    # Find all depth files in the processed directory (recursively)
    # Looking for .npy files in 'depth' subdirectories
    search_pattern = os.path.join(input_dir, '**', 'depth', '*.npy')
    depth_files = glob.glob(search_pattern, recursive=True)
    
    if not depth_files:
        print(f"No depth files found in {input_dir}. Did you run prepare_dataset.py?")
        return

    print(f"Generating labels for {len(depth_files)} depth maps...")
    
    for depth_path in tqdm(depth_files):
        # Load Depth
        depth = np.load(depth_path)
        
        # Normalize depth if needed (assuming raw values, roughness depends on scale)
        # For generalization, let's normalize to 0-255 range for std calculation or keep as is?
        # Keeping as is preserves physical meaning if calibration is consistent.
        # If float 0-1 (from resize?), scale up.
        
        # Calculate Roughness
        roughness = compute_roughness(depth.astype(np.float32))
        
        # Create Binary Mask
        # Safe (1) if roughness < threshold
        # Unsafe (0) otherwise
        mask = np.zeros_like(roughness, dtype=np.uint8)
        mask[roughness < rough_threshold] = 255
        
        # Construct Output Path
        # Input: .../WildUAV_Processed/Mapping/seq00/depth/000000.npy
        # Output: .../WildUAV_Processed/Mapping/seq00/labels/000000.png
        
        rel_path = os.path.relpath(depth_path, input_dir)
        save_rel = rel_path.replace('depth', 'labels').replace('.npy', '.png')
        save_path = os.path.join(output_dir, save_rel)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Safe Landing Zones from Depth")
    parser.add_argument("--input", type=str, default="../data/WildUAV_Processed", help="Path to processed dataset")
    parser.add_argument("--output", type=str, default="../data/WildUAV_Processed", help="Path to save labels (usually same root)")
    parser.add_argument("--threshold", type=float, default=10.0, help="Roughness threshold (lower = stricter safety)")
    
    args = parser.parse_args()
    
    generate_labels(args.input, args.output, args.threshold)
