import cv2
import numpy as np
from PIL import Image


def main():
    img = cv2.imread("./123508885.jpeg")
    red = np.full(img.shape, (0,0,255), np.uint8)
    red_tinted = cv2.addWeighted(img, 0.7, red, 0.3, 0)
    cv2.imshow('Red Isolated Image', red_tinted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pass

if __name__ == "__main__":
    main()
