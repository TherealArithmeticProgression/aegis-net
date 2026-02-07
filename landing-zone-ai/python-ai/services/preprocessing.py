from torchvision import transforms
from PIL import Image

def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_file)
    return transform(image).unsqueeze(0)
