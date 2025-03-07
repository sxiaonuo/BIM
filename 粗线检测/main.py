
import os

import cv2
from PIL import Image
import random
import numpy as np
from tqdm import tqdm

from utils.detect_line import detect_lines


# 获取轮廓
def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nowhite = np.where(binary != 255)
    nowhite = set(zip(*nowhite))
    img_padding = np.zeros((img.shape[0] + 1, img.shape[1] + 1, 3), dtype=np.uint8)
    img_padding[:-1, :-1, :] = img

    white = np.ones_like(img, dtype=np.uint8 ) * 255
    for x, y in tqdm(nowhite):
        for dx, dy in zip([0, 1, 0, -1], [1, 0, -1, 0]):
            if (x + dx, y + dy) not in nowhite or (img_padding[x + dx, y + dy] != img_padding[x, y]).any():
                white[x, y] = img[x, y]
    return white


if __name__ == '__main__':

    img = cv2.imread("../static/img/b1.png")

    contours_img = get_contours(img)
    os.makedirs("workdir/run", exist_ok=True)
    Image.fromarray(contours_img).save("workdir/run/b1f.png")