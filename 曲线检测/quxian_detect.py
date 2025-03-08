import os
import random

import cv2
from PIL import Image
from utils.detect_line import *

def draw_detect_line():
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pt1 = line[0]
        pt2 = line[1]
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'randomcolor.png')
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = line[2]
        pt1 = line[0]
        pt2 = line[1]
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'originalcolor.png')


if __name__ == '__main__':
    # 读取配置
    cfg = get_cfg_defaults()
    cfg.SAVE_DIR = f'workdir/run/{random.randint(0, 100000):06d}/'
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(cfg)

    ori_img = cv2.imread("../src/1.png")
    ori_img = ori_img[3000:-3000, 3000:-3000, :]  # 为了更快看到结果，只截取一部分
    Image.fromarray(ori_img).save(cfg.SAVE_DIR + 'ori.png')

    lines, total_lines, all_elines = detect_lines(ori_img, cfg)

    # draw_detect_line()

    for line in lines:
        line[0]
