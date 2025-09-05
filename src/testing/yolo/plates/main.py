from red_color import red_transform
from PIL import Image
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor
import os

path_img = "/home/espala/Dataset/plates/train/images"
path_label = path_img.split("/")[:-1] 
path_label = "/".join(path_label) + "/labels"


img_dict = {}
name_img = []

def get_data():
    global img_dict, path_img
    img_dict = {f"{img}": f'{img.split(".jpg")[0] + ".txt"}' for img in os.listdir(path_img) if img.endswith(".jpg")}
    
def process_one(args):
    img_name, names, i, path_img, path_label = args
    if not names.startswith('$%@'):
        with open(path_label + "/" + names) as f:
            linha = f.readlines()[0]
            lined = linha.strip().split()
            if len(lined) != 5:
                raise ValueError(f"Arquivo {names} tem {len(lined)} valores: {lined}")
            c, x_c, y_c, w, h = lined
        img =  red_transform(
            path_img + "/" + img_name,
            path_img + "/" + f"{i}img.jpg",
            float(x_c), float(y_c), float(w), float(h)
        )
        
        with open(path_label + "/" + f"{i}img.txt", "w") as f:
            f.write(f"{c} {x_c} {y_c} {w} {h}")

        return img
def data_aug():
    global img_dict, path_img, path_label
    size = len(img_dict)

    # Criar um array numpy de argumentos
    args_array = np.array([
        (img_name, names, i, path_img, path_label)
        for i, (img_name, names) in enumerate(img_dict.items())
    ], dtype=object)

    with ProcessPoolExecutor() as executor:
        for j, _ in enumerate(executor.map(process_one, args_array), 1):
            print(f"Progress {j/size:.2%}", end="\r")
    pass

def delete_augmented_data():
    global img_dict, path_img, path_label
    size = len(name_img)
    i = 0
    for img in name_img():
        try:
            os.remove(path_img + "/" + img)
            i += 1
            print(f"{float(i/size)}%", end="\r")
        except:
            print(end="")

def train():
    global path_img
    path_config = path_img.split("/")[1:-2]
    path_config = "/" + "/".join(path_config)
    model = YOLO("yolov8n.pt")
    res = model.train(
        data = f'{path_config}/data.yaml',
        epochs = 50,
        imgsz=640,
        device='cuda',
        patience=8,
        verbose=True
    )

    model.export(format='onnx')


if __name__ == "__main__":
    path_config = path_img.split("/")[1:-2]
    path_config = "/" + "/".join(path_config)
    
    for x in os.listdir(path_img):
        if 'img' in x:
            print("Removing", x, end="\r")
            os.remove(path_img + "/" + x)

    for x in os.listdir(path_label):
        if 'img' in x:
            print("Removing", x, end="\r")
            os.remove(path_label + "/" + x)
    
    get_data()
    data_aug()
    
    train()

    delete_augmented_data()
    
    
#ls Dataset/plates/train/images/ | grep "img"