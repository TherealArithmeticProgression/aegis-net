# Model Design

## Model Architecture
- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Head**: Custom fully connected layers for regression (safety score) and classification (terrain type).

## Heatmap Generation
- Uses Grad-CAM to visualize attention on the input image.

## Training Data
- Synthetic and real-world aerial imagery of landing zones.
