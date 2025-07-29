import dotenv
import os
from ultralytics import YOLO


def pop_yaml():
    dotenv.load_dotenv()
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
    model = YOLO("yolov8l.pt")
    res = model.train(
        data = 'config.yaml',
        epochs = 100,
        imgsz=640,
        device='cuda'
    )
    metrics = model.val()
    results = model("./teste1.jpg")
    results[0].show()

    model.export(format='onnx')

if __name__ == "__main__":
    pop_yaml()
    main()
    #clean_yaml()
    
