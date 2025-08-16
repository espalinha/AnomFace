from torch.utils.data import dataloader, Dataset
import torchvision.transforms as T
from src.plates.yolo.red_filter import RedTintTransform
from dotenv import load_dotenv
import os

class PlatesDataset(Dataset):
    def __init__(self):
        load_dotenv()
        self.path = os.getenv("plates")
        self.transform = T.Compose([
            RedTintTransform(),
            T.ToTensor(),  # Convert to tensor for DataLoader
        ])

    """
        Implement the rest of the dataset, not for yolo, for normal models
    """
    

    
