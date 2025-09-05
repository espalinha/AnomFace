from PIL import Image

def convertRGB(img: Image.Image) -> Image.Image: 
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img

def open_img(img_path) -> Image.Image:
    try:
        
        img = Image.open(img_path)
        return img
    except FileNotFoundError:
        print("error, FileNotFound")
        return 1

import numpy as np

def apply_red_region(img: Image.Image, x_c: float, y_c: float, w: float, h: float) -> Image.Image:
    w_img, h_img = img.size
    arr = np.array(img)

    # Converter YOLO -> coordenadas em pixels
    x_center = int(x_c * w_img)
    y_center = int(y_c * h_img)
    w_box = int(w * w_img)
    h_box = int(h * h_img)

    x1 = max(0, x_center - w_box // 2)
    y1 = max(0, y_center - h_box // 2)
    x2 = min(w_img, x_center + w_box // 2)
    y2 = min(h_img, y_center + h_box // 2)

    # aplicar filtro só dentro da região com slicing
    region = arr[y1:y2, x1:x2, :]

    region = region.astype(np.float32)
    region[..., 0] = np.clip(region[..., 0] * 1.5, 0, 255)  # aumentar R
    region[..., 1] = region[..., 1] * 0.5                   # reduzir G
    region[..., 2] = region[..., 2] * 0.5                   # reduzir B
    arr[y1:y2, x1:x2, :] = region.astype(np.uint8)

    return Image.fromarray(arr)


def red_transform(img_path, save, x_c, y_c, w, h):
    img = open_img(img_path)
    
    if img == 1:
        return
    img = convertRGB(img)

    # aplicar só na região (convertendo de YOLO -> pixels)
    img = apply_red_region(img, x_c, y_c, w, h)    

    img.save(save)
    return img

if __name__ == "__main__":
    # exemplo: bounding box YOLO (x_c, y_c, w, h) normalizados
    red_transform(
        "/home/espala/AnomFace-main/src/testing/yolo/plates/teste1.jpg",
        "/home/espala/AnomFace-main/src/testing/yolo/plates/teste1_red.jpg",
        0.5, 0.5, 0.3, 0.2   # centro no meio da imagem, box 30% largura, 20% altura
    )
