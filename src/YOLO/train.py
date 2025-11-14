from ultralytics import YOLO
import torch.nn as nn
import os

def train():
    model = YOLO('yolo11l.pt')
    model.train(
        data="config.yaml",  # path to dataset config file
        epochs=100,          # number of training epochs
        imgsz=640,          # training image size
        amp=True,
        batch=16,           # batch size
        device=[0],
        project='/home/espala/AnomFace-main/src/runs_project',
        name='train_yolo',
        patience=5
    )

if __name__ == "__main__":
    train()
