import sys
import os
from numpy.random import random
from data_load import get_dataloader
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class AnchorBox(nn.Module):
    pass


def draw_boxes(image, boxes):
    # Converta o tensor de volta para PIL Image se necessário
    if isinstance(image, torch.Tensor):
        to_pil = T.ToPILImage()
        image = to_pil(image)
    
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # Certifique-se de que as coordenadas da caixa são números
        x, y, w, h = map(int, box)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    return image


def main(type="train"):
    dataloader = get_dataloader()
    
    
    for batch in dataloader:
        tensors, pil_images, names, boxes_list = batch
        
        for tensor, pil_img, name, boxes in zip(tensors, pil_images, names, boxes_list):
            if random() > 0.97:
                
                image_with_boxes = draw_boxes(pil_img, boxes)
                image_with_boxes.show()
            
            print(f"{name}:", boxes)


if __name__ == "__main__":
    main()
