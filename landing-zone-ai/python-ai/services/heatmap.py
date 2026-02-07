import numpy as np
import cv2
import os
import uuid
import matplotlib.cm as cm

def generate_heatmap(mean_pred):
    """
    Generates a visual heatmap from the mean prediction (H, W).
    """
    # 1. Normalize to 0-255
    heatmap_norm = (mean_pred * 255).astype(np.uint8)
    
    # 2. Apply Colormap (JET is common for heatmaps)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # 3. Save
    from config import Config
    output_dir = Config.HEATMAP_FOLDER
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    filename = f"heatmap_{uuid.uuid4().hex}.png"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, heatmap_color)
    
    # Return endpoint relative URL for the client
    # Client accesses: http://localhost:3000/heatmaps/filename.png
    return f"/heatmaps/{filename}"
