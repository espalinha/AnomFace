import cv2
import torchvision.transforms as T
from PIL import Image
import numpy as np

class RedTintTransform():
    def __init__(self, alpha=0.7, beta=0.3):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        # Create red tint layer
        red = np.full(img_np.shape, (0, 0, 255), dtype=np.uint8)
        # Apply weighted addition
        red_tinted = cv2.addWeighted(img_np, self.alpha, red, self.beta, 0)
        # Convert back to PIL Image
        return Image.fromarray(red_tinted)



