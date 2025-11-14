from ultralytics import RTDETR
import torch.nn as nn
import os

def train():
    model = RTDETR('rtdetr-l.pt')
    model.train(
        data="config.yaml",  # path to dataset config file
        epochs=100,          # number of training epochs
        imgsz=640,          # training image size
        #amp=True,
        batch=3,           # batch size
        device=[0, 1, 2],
        project='/home/espala/AnomFace-main/src/runs_project',
        name='train_rtdetr',
        patience=15
    )

if __name__ == "__main__":
    train()
