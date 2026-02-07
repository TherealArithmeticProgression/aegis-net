import os

class Config:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/landing_model.pth')
    UPLOAD_FOLDER = 'uploads'
    HEATMAP_FOLDER = 'heatmaps'
