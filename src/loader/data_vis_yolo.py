import os
import dotenv
from PIL import Image
import random
from PIL import ImageDraw

dotenv.load_dotenv()

class Visualization():
    def __init__(self):
        self.img_path = os.getenv("TRAIN_YOLO")
        self.anom_path = os.getenv("TRAIN_ANNOT_YOLO")
        self.img_files = [f for f in os.listdir(self.img_path) if f.endswith('.jpg')]
        self.annotations = self._load_anno()

    def _load_anno(self):
        global_dict = {}
        for img_name in self.img_files:
            anno_file = os.path.join(self.anom_path, img_name.replace('.jpg', '.txt'))
            if os.path.exists(anno_file):
                with open(anno_file, 'r') as f:
                    boxes = [line.strip().split()[1:] for line in f.readlines()]
                global_dict[img_name] = boxes
        
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
        for img_name in self.img_files:
            if random.random() > 0.9998:
                img_path = os.path.join(self.img_path, img_name)
                pil_image = Image.open(img_path).convert('RGB')
                boxes = self.annotations.get(img_name, [])
                
                if boxes:
                    image_with_boxes = self._draw_boxes(pil_image, boxes)
                    image_with_boxes.show()
                else:
                    print(f"No annotations for {img_name}")
    
if __name__ == "__main__":
    dataset = Visualization()
    dataset.draw()

