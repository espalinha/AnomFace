from src.loader import data_load as dl
from numpy.random import random
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw



def main():
    print("Loading YOLO first Example...")
    dataloader = dl.get_dataloader()
    for batch in dataloader:
        tensors, pil_images, names, boxes_list = batch
        
        for tensor, pil_img, name, boxes in zip(tensors, pil_images, names, boxes_list):
            print("Sim")

if __name__ == "__main__":
    main()

    

