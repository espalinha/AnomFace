from dotenv import load_dotenv
import os
import torchvision.transforms as T
from src.plates.yolo.red_filter import RedTintTransform
from src.plates.yolo.dataloader import PlatesDataset
from ultralytics import YOLO

def pop_yaml():
    load_dotenv()
    path_ = os.getenv("PATH_")
    with open("config.yaml", "w") as f:
        f.write(f"train: {path_}images/train\n")
        f.write(f"val: {path_}images/val\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['face']\n")
    pass

def clean_yaml():
    with open("config.yaml", "w") as f:
        f.write("")
    pass

def main():
    
    model = YOLO("yolov8n.pt")
    res = model.train(
        data = 'config.yaml',
        epochs = 100,
        imgsz=640,
        device='cuda'
    )
    metrics = model.val()
    model.export(format='onnx')

if __name__ == "__main__":
    clean_yaml()
    pop_yaml()
    main()




