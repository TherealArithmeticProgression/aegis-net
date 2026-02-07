from torchvision import transforms
from PIL import Image
import torch

def preprocess_image(image_file):
    """
    Prepares an image for inference:
    1. Resize to model input size (256x256).
    2. Convert to Tensor (scales to 0-1).
    3. Normalize (using standard ImageNet mean/std).
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Handle both file paths and PIL images
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB')
        
    return transform(image).unsqueeze(0)

