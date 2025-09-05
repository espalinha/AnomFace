import os
import shutil

path = "/home/espala/Dataset/plates"
path_out = "/home/espala/AnomFace-main/data/widerface_plate/dataset"
prefix_plate = "plates_"

def copy_images():
    dicio = {"train": "train", "valid": "train", "test": "val"}
    for key in dicio.keys():
        path_in = os.path.join(path, key)
        path_out_ = os.path.join(path_out, dicio[key])
        print("saida:",path_out_)
        images_in = os.path.join(path_in, "images")
        labels_in = os.path.join(path_in, "labels")
        images_out = os.path.join(path_out_, "image")
        labels_out = os.path.join(path_out_, "labels")
        #os.makedirs(images_out, exist_ok=True)
        #os.makedirs(labels_out, exist_ok=True)
        for idx, file in enumerate(os.listdir(images_in)):
            if (
                
                    file.endswith(".jpg")
                    or file.endswith(".png")
                    or file.endswith(".jpeg")
                
            ):  
                
                src_img = os.path.join(images_in, file)
                new_name = f"{prefix_plate}{idx}{os.path.splitext(file)[1]}"
                dst_img = os.path.join(images_out, new_name)
                shutil.copy(src_img, dst_img)
                
                
                # Processa o label correspondente, se existir
                label_name = os.path.splitext(file)[0] + ".txt"
                src_label = os.path.join(labels_in, label_name)
                dst_label = os.path.join(labels_out, os.path.splitext(new_name)[0] + ".txt")
                
                print(f"Copying {dst_img} to {dst_label}")
                

                
                if os.path.exists(src_label):
                    with open(src_label, "r") as f:
                        lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = "1"
                            new_lines.append(" ".join(parts))
                    
                    with open(dst_label, "w") as f:
                        for l in new_lines:
                            f.write(l + "\n")
                


copy_images()