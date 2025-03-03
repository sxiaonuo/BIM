import cv2
from PIL import Image, ImageDraw
import numpy as np
import random
import os
import math

from tqdm import tqdm
from tqdm.contrib import itertools

from yacs.config import CfgNode as CN

if __name__ == '__main__':
    import utils.detect_line as detect_line

    ori_img = cv2.imread('../static/img/b1.png')
    ori_img = ori_img[5000:18000, 5000:17000, :]
    lines, _ = detect_line.detect_lines(ori_img)

    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(white, line[0], line[1], color, 1)
    Image.fromarray(white).save("workdir/run/random_color_line.png")
    white = np.ones_like(ori_img) * 255
    for line in lines:
        cv2.line(white, line[0], line[1], line[2], 1)
    Image.fromarray(white).save("workdir/run/line.png")
    Image.fromarray(white).show()