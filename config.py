import os
import sys

def dotenv_loader(path):
    with open("./src/loader/.env", "w") as f:
        f.write(f'TRAIN={os.path.join(path,"WIDER_train/images/0--Parade")}')
        f.write(f'TEST={os.path.join(path,"WIDER_test/images/0--Parade")}') 
        f.write(f'TRAIN_ANNOT={os.path.join(path,"wider_face_split/wider_face_train_bbx_gt.txt")}')
        f.write(f'VAL_ANNOT={os.path.join(path,"wider_face_split/wider_face_val_bbx_gt.txt")}') 
        f.write(f'TEST_ANNOT={os.path.join(path,"wider_face_split/wider_face_test_bbx_gt.txt")}')
        f.write(f'PRE={os.path.join(path)}') 
        f.write(f'PRE_TRAIN={os.path.join(path,"WIDER_train/images/")}') 
        f.write(f'PRE_VAL={os.path.join(path,"WIDER_val/images/")}') 

def dotenv_yolo(path):
    with open("./src/yolo/.env", "w") as f:
        f.write(f"PATH_={os.path.join(path, 'dataset')}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data"
    dotenv_loader(path)
    dotenv_yolo(path)

if __name__ == "__main__":
    main()
