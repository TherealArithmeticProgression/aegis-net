import os
import torch
import numpy as np
import cv2
from PIL import Image
from .preprocessing import preprocess_image
from .heatmap import generate_heatmap
from config import Config
from models.yolov8_landing import YOLOv8LandingZone, YOLOv8SegmentationWrapper


class TTAAugmentor:
    """
    Test-Time Augmentation for uncertainty estimation.
    Runs inference on flipped and scaled versions of the image.
    """
    def __init__(self, scales=[0.75, 1.0, 1.25], flips=[False, True]):
        self.scales = scales
        self.flips = flips
    
    def generate_augmented_images(self, image):
        """
        Generate augmented versions of the image.
        
        Args:
            image: numpy array (H, W, 3)
        
        Returns:
            List of (augmented_image, reverse_transform_fn) tuples
        """
        augmented = []
        original_size = (image.shape[1], image.shape[0])  # (W, H)
        
        for scale in self.scales:
            for flip in self.flips:
                aug_img = image.copy()
                
                # Scale
                if scale != 1.0:
                    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                    aug_img = cv2.resize(aug_img, new_size, interpolation=cv2.INTER_LINEAR)
                
                # Horizontal flip
                if flip:
                    aug_img = cv2.flip(aug_img, 1)
                
                # Store with reverse transform info
                augmented.append({
                    'image': aug_img,
                    'scale': scale,
                    'flipped': flip,
                    'original_size': original_size
                })
        
        return augmented
    
    def reverse_transform(self, prediction, scale, flipped, original_size):
        """
        Reverse the augmentation on the prediction.
        """
        # Reverse flip
        if flipped:
            prediction = cv2.flip(prediction, 1)
        
        # Reverse scale (resize back to original)
        prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_LINEAR)
        
        return prediction


class InferenceService:
    def __init__(self, use_wrapper=True):
        self.use_wrapper = use_wrapper
        self.tta = TTAAugmentor()
        
        if use_wrapper:
            self.model = YOLOv8SegmentationWrapper()
            print("YOLOv8 Segmentation Wrapper loaded.")
        else:
            self.model = YOLOv8LandingZone(pretrained=True)
            model_path = Config.MODEL_PATH.replace('landing_model.pth', 'yolo_landing.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("Custom YOLOv8 model loaded.")
            else:
                print("Warning: Custom weights not found, using pretrained.")
            self.model.eval()

    def predict(self, image_file, use_tta=True):
        """
        Runs inference with Test-Time Augmentation (TTA) for uncertainty estimation.
        
        Instead of MC Dropout, we run inference on flipped and scaled versions.
        Uncertainty = variance across augmented predictions.
        """
        # Load image
        if isinstance(image_file, str):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif hasattr(image_file, 'read'):
            image = np.array(Image.open(image_file))
        else:
            image = np.array(image_file)
        
        if use_tta:
            # Generate augmented versions
            augmented_images = self.tta.generate_augmented_images(image)
            predictions = []
            
            for aug_data in augmented_images:
                # Run prediction on augmented image
                if self.use_wrapper:
                    pred = self.model.predict(aug_data['image'])
                else:
                    # Custom model inference
                    input_tensor = preprocess_image(Image.fromarray(aug_data['image']))
                    self.model.eval()
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                
                # Reverse transform to align with original image
                pred = self.tta.reverse_transform(
                    pred, 
                    aug_data['scale'], 
                    aug_data['flipped'],
                    aug_data['original_size']
                )
                predictions.append(pred)
            
            # Calculate mean and variance across TTA predictions
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)
            variance = np.var(predictions, axis=0)
        else:
            # Single prediction without TTA
            if self.use_wrapper:
                mean_pred = self.model.predict(image)
            else:
                input_tensor = preprocess_image(Image.fromarray(image))
                self.model.eval()
                with torch.no_grad():
                    output = self.model(input_tensor)
                    mean_pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            variance = np.zeros_like(mean_pred)
        
        # Calculate Confidence Score
        confidence_map = mean_pred * (1 - variance)
        global_score = float(np.mean(confidence_map))
        
        # Generate Heatmap with overlay
        heatmap_path = generate_heatmap(mean_pred, variance, image)
        
        return {
            "score": global_score,
            "heatmapUrl": heatmap_path,
            "stats": {
                "mean_variance": float(np.mean(variance)),
                "max_confidence": float(np.max(confidence_map)),
                "n_augmentations": len(self.tta.scales) * len(self.tta.flips) if use_tta else 1
            }
        }

    def predict_simple(self, image_file):
        """Single forward pass inference (no TTA)."""
        return self.predict(image_file, use_tta=False)
