#Here I will write the code to preprocess the data for the yolo learning
import os
from os.path import isdir
import shutil
from typing import override
fixed=""

def _create_fixed_dir(path):
    global fixed
    fixed = os.path.join(path, "dataset")
    os.makedirs(os.path.join(path, "dataset"), exist_ok=True)

class BoxesStruct:
    def __init__(self, path, boxes):
        self.path = path
        self.boxes = boxes

    def __str__(self):
        return f"{self.path}, {self.boxes}"

    def get_box(self):
        return self.boxes
    
    def get_path(self):
        return self.path

def _load_annotations(envname: str):
    import dotenv
    dotenv.load_dotenv()
    annot_paths = os.getenv(envname)
    global_dict = {}
    with open(annot_paths, "r") as fl:
            lines = fl.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip() #Only letters
                if line.endswith('.jpg'):
                    img_name = line.split("/")[-1] #take the name
                    num_boxes = int(lines[i+1].strip()) #always letting only the letters
                    boxes = []
                    for j in range(num_boxes):
                        box_line = lines[i+2+j].strip() #we only wnat the first four elements 78 221 7 8
                        box = list(box_line.split()[:4])
                        boxes.append(box)
                    gg = BoxesStruct(line, boxes)
                    
                    
                    global_dict[img_name] = gg
                    i += 2 + num_boxes
                else:
                    i += 1
    return global_dict

from PIL import Image

def annotations_to_yolo_format(boxes_struct: BoxesStruct, name):
    #boxes = [x, y, w, h]
    boxes = boxes_struct.get_box()
    path = boxes_struct.get_path()

    img__ = os.getenv(f"PRE_{name}")

    imga = Image.open(os.path.join(img__, path))
    img_w, img_h = imga.size


    yolo_boxes = []
   
    #print(boxes)
    for box in boxes:
        w = float(box[2])
        h = float(box[3])
        #print(h)
        x_center = ((float(box[0]) + w) / 2)/img_w
        y_center = ((float(box[1]) + h) / 2)/img_h
        yolo_boxes.append([x_center, y_center, w/img_w, h/img_h])
    return yolo_boxes
"""
def create_train_dir(fixed):
    
    dest_train_path = os.path.join(fixed, "images", "train")
    os.makedirs(dest_train_path, exist_ok=True)
    annotation = os.path.join(fixed, "labels", "train")
    os.makedirs(os.path.join(annotation), exist_ok=True)
    return dest_train_path, annotation
"""

def _create_dir(fixed, name):
    dest_path = os.path.join(fixed, "images", f"{name}")
    os.makedirs(dest_path, exist_ok=True)
    annotation = os.path.join(fixed, "labels", f"{name}")
    os.makedirs(os.path.join(annotation), exist_ok=True)
    return dest_path, annotation
   
def _get_image_label(path, name: str, i: int):
    if fixed == "":
        _create_fixed_dir(path)
 
    dest_path, annotation = _create_dir(fixed, name.lower())
    global_dict = _load_annotations(f"{name}_ANNOT")
    
    src_path = os.getenv(f"PRE_{name}")
    #i = 0
      

    print(name)
    for dir in os.listdir(src_path):
        for file in os.listdir(os.path.join(src_path, dir)):
            #print(file)
            
            annot = annotations_to_yolo_format(global_dict[file], name)
            #print(annot)
            
            if file.endswith(".jpg"):
                src_file = os.path.join(src_path, dir, file)
                dest_file = os.path.join(dest_path)
                shutil.copy(src_file, dest_file)
                
                with open(os.path.join(annotation, f"img{i}.txt"), "w") as f:
                    for box in annot:
                        f.write(f"{0} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                os.rename(os.path.join(dest_file, file), os.path.join(dest_file, f"img{i}.jpg"))
                i += 1
            
#                     print(f"Copied {file} to {dest_file}")
    
    return i
"""
def _get_train_image_label(path):
    if fixed == "":
        _create_fixed_dir(path)

    dest_train_path, annotation = create_train_dir(fixed)
    global_dict = _load_annotations("TRAIN_ANNOT")
    
    src_train_path = os.getenv("PRE_TRAIN")
    i = 0
      
    for dir in os.listdir(src_train_path):
        for file in os.listdir(os.path.join(src_train_path, dir)):
            #print(file)
            annot = annotations_to_yolo_format(global_dict[file])
            #print(annot)
            
            if file.endswith(".jpg"):
                src_file = os.path.join(src_train_path, dir, file)
                dest_file = os.path.join(dest_train_path)
                shutil.copy(src_file, dest_file)
                
                with open(os.path.join(annotation, f"img{i}.txt"), "w") as f:
                    for box in annot:
                        f.write(f"{0} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                os.rename(os.path.join(dest_file, file), os.path.join(dest_file, f"img{i}.jpg"))
                i += 1
            
#                     print(f"Copied {file} to {dest_file}")
    print(os.listdir(dest_train_path))
"""
"""
def create_val_dir(fixed):
    
    dest_train_path = os.path.join(fixed, "images", "val")
    os.makedirs(dest_train_path, exist_ok=True)
    annotation = os.path.join(fixed, "labels", "val")
    os.makedirs(os.path.join(annotation), exist_ok=True)
    return dest_train_path, annotation

def _get_val_image_label(path):
    if fixed == "":
        _create_fixed_dir(path)

    dest_val_path, annotation = create_val_dir(fixed)
    global_dict = _load_annotations("VAL_ANNOT")
    
    src_val_path = os.getenv("PRE_VAL")
    i = 0
      
    for dir in os.listdir(src_val_path):
        for file in os.listdir(os.path.join(src_val_path, dir)):
            #print(file)
            annot = annotations_to_yolo_format(global_dict[file])
            #print(annot)
            
            if file.endswith(".jpg"):
                src_file = os.path.join(src_val_path, dir, file)
                dest_file = os.path.join(dest_val_path)
                shutil.copy(src_file, dest_file)
                
                with open(os.path.join(annotation, f"img{i}.txt"), "w") as f:
                    for box in annot:
                        f.write(f"{0} {box[0]} {box[1]} {box[2]} {box[3]}\n")
                os.rename(os.path.join(dest_file, file), os.path.join(dest_file, f"img{i}.jpg"))
                i += 1
            
#                     print(f"Copied {file} to {dest_file}")
    print(os.listdir(dest_val_path))
"""
def fix_data(path):
    i = 0
    _create_fixed_dir(path)
    i = _get_image_label(path, "TRAIN", i)
    _get_image_label(path, "VAL", i)
    #print(fixed)
    pass

if __name__ == "__main__":
    # This is the main function that will be called when the script is run
    # It will call the fix_data function to preprocess the data
    import dotenv
    dotenv.load_dotenv()
    path = os.getenv("PRE")
    fix_data(path)
    print("Data preprocessing complete.")


