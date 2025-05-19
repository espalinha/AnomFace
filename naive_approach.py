"""
Matheus Espalaor
matheusespalaor14@gmail.com

In this project, we implemented a naive approach to face recognition as an initial step toward building a functional system. 
At this stage, the focus was not on optimizing the model, but rather on achieving a working version capable of performing inference. 
The goal was to validate the core functionality and establish a solid foundation for future improvements.
"""

"""
@inproceedings{yang2016wider,
	Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
	Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	Title = {WIDER FACE: A Face Detection Benchmark},
	Year = {2016}}

We will use the WIDERFACE Dataset: http://shuoyang1213.me/WIDERFACE/ from Yang Shou
And for training and test, for first, we will use the .txt files provided by them, that select images for training and testing.
"""

import sys
import os

"""
Input: dir path, type that we want (train, test, validation (val))
Return: will return a tuple: the dataset of the type and the annotation
(if test will not have annotation)
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#for now, working only for 1 folder
class WiderFaceDataset(Dataset):
    def __init__(self, img_path, anom_path):
        self.img_path = img_path
        self.anom_path = anom_path
        self.img_files = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]
        self.annotations = self.load_anno()
        self.to_tensor = transforms.ToTensor()

    def load_anno(self):
        global_dict = {}
        with open(self.anom_path, "r") as fl:
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
                    global_dict[img_name] = boxes
                    i += 2 + num_boxes
                else:
                    i += 1
        return global_dict


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = self.img_files[index]
        img_path = os.path.join(self.img_path, img_name)
        image = self.to_tensor(Image.open(img_path).convert('RGB'))
        boxes = self.annotations.get(img_name, [])
        return image, self.img_files[index], boxes




def custom_collate_fn(batch):
    images, name, boxes = zip(*batch)
    return list(images), list(name),list(boxes)


#path will be (train, test, val) val is optional
def main(path, type="train"):
    train_test = "0--Parade" #For now, I just will use one type of image.
    dir_name = f"{path}/WIDER_{type}/images/{train_test}"
    #wider_face_train_bbx_gt
    annotation = f"{path}/wider_face_split/wider_face_{type}_bbx_gt.txt"

    dataset = WiderFaceDataset(dir_name, annotation)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    for img, name, boxes in dataloader:
        print(f"{name}:", boxes)

if __name__ == "__main__":
    #dir path that contains the dataset, with the real names
    path = sys.argv[1]
    main(path)



        


