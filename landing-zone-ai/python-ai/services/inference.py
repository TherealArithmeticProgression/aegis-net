import os
import torch
import numpy as np
from .preprocessing import preprocess_image
from .heatmap import generate_heatmap
from config import Config
from models.unet_resnet import ResNetUNet

class InferenceService:
    def __init__(self):
        self.model = ResNetUNet(n_class=1)
        if os.path.exists(Config.MODEL_PATH):
           self.model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device('cpu')))
           print("Model loaded successfully.")
        else:
           print("Warning: Model weights not found. Running with random weights.")
        self.model.eval()

    def predict(self, image_file, mc_samples=10):
        """
        Runs Monte Carlo Dropout inference.
        Returns:
            - heatmapUrl: path to saved visualized heatmap
            - score: single float confidence score
            - stats: dictionary with detailed metrics
        """
        # Preprocess
        input_tensor = preprocess_image(image_file) # (1, 3, 256, 256)
        
        # MC Dropout Loop
        predictions = []
        self.model.train() # Enable Dropout
        
        with torch.no_grad():
            for _ in range(mc_samples):
                output = self.model(input_tensor)
                output = torch.sigmoid(output) # Sigmoid for probability (0-1)
                predictions.append(output.cpu().numpy())
        
        # Calculate Mean and Variance
        predictions = np.array(predictions) # (N, 1, 1, H, W)
        mean_pred = np.mean(predictions, axis=0)[0, 0] # (H, W)
        variance = np.var(predictions, axis=0)[0, 0]   # (H, W)
        
        # Calculate Confidence Score
        # Formula: Mean * (1 - Variance)
        # Note: High variance reduces confidence.
        # We compute a pixel-wise confidence map, then aggregate.
        confidence_map = mean_pred * (1 - variance)
        
        # Aggregate logic (e.g., take the max confidence area or average of safe zones)
        # For simplicity, let's take the average of the top 20% confident pixels as the global score
        # or just the mean of the map.
        # Let's use the mean of the confidence map for the global score.
        global_score = float(np.mean(confidence_map))
        
        # Generate Heatmap Image (Visualizing Mean Prediction)
        heatmap_path = generate_heatmap(mean_pred) # You need to implement/update this to handle numpy array
        
        return {
            "score": global_score,
            "heatmapUrl": heatmap_path,
            "stats": {
                "mean_variance": float(np.mean(variance)),
                "max_confidence": float(np.max(confidence_map)),
            }
        }
