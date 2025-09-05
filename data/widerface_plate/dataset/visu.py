import os
import dotenv
from PIL import Image
import random
from PIL import ImageDraw

dotenv.load_dotenv()

class Visualization():
    def __init__(self):
        self.img_path = [
            "/home/espala/AnomFace-main/data/widerface_plate/dataset/train/images", 
            "/home/espala/AnomFace-main/data/widerface_plate/dataset/val/images"
            ]
        self.anom_path = [
            "/home/espala/AnomFace-main/data/widerface_plate/dataset/train/labels", 
            "/home/espala/AnomFace-main/data/widerface_plate/dataset/val/labels"
            ]
        self.img_files = [os.path.join(ipg, f) for ipg in self.img_path for f in os.listdir(ipg) if f.endswith('.jpg')]
        self.annotations = self._load_anno()

    def _load_anno(self):
        global_dict = {}
        for img_path in self.img_files:
            if self.img_path[0] in img_path:
                label_dir = self.anom_path[0]
            else:
                label_dir = self.anom_path[1]
            img_name = os.path.basename(img_path)
            anno_file = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(anno_file):
                with open(anno_file, 'r') as f:
                    boxes = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            # Ignora o class_id
                            boxes.append(parts[1:])
                global_dict[img_path] = boxes
        return global_dict
    
    def _draw_boxes(self, image, boxes):
        
        draw = ImageDraw.Draw(image)
        size = image.size
        for box in boxes:
            x, y, w, h = map(float, box)
            x = x * size[0]
            y = y * size[1]
            w = w * size[0]
            h = h * size[1]
            draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=2)
        return image

    def draw(self):
        # Cria a pasta 'ret' se não existir
        ret_dir = os.path.join(os.path.dirname(self.img_path[0]), 'ret')
        os.makedirs(ret_dir, exist_ok=True)
        for img_path in self.img_files:
            # Apenas desenha se o número aleatório for maior que 0.9999
            if random.random() > 0.9999:
                pil_image = Image.open(img_path).convert('RGB')
                boxes = self.annotations.get(img_path, [])
                if boxes:
                    image_with_boxes = self._draw_boxes(pil_image, boxes)
                    # Salva a imagem na pasta 'ret' com o mesmo nome do arquivo original
                    out_path = os.path.join(ret_dir, os.path.basename(img_path))
                    image_with_boxes.save(out_path)
                else:
                    print(f"No annotations for {img_path}")
    
if __name__ == "__main__":
    dataset = Visualization()
    dataset.draw()

