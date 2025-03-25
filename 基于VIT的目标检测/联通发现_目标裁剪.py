import os

import cv2
from PIL import Image
import random
import numpy as np
from utils.detect_line import detect_lines
from utils.detect_connect import detect_connect, get_max_coord, get_min_coord
from utils.hot import hot_detect_line, hot_dectect_connect
from tqdm import tqdm

import json

def color_filter(lines, color):
    """颜色过滤"""
    return [line for line in lines if line[2] != color]

if __name__ == '__main__':
    """超参数"""
    img_name = 4
    padding = 20

    """"""
    # 连通发现
    ori_img = cv2.imread(f'../static/img/{img_name}.png')
    # groups = hot_dectect_connect(f'../static/img/{img_name}.png', None, None, 0, path='../static/record/')
    # groups = [group for group in groups if group[0][2] != (0, 0, 0) and len(group) >= 4]
    groups = json.load(open("group_4.json"))
    print(len(groups))

    # 裁图
    # 记录坐标
    group_coords = []
    os.makedirs(f"workdir/datasets/{img_name}", exist_ok=True)
    for i, group in enumerate(tqdm(groups)):
        max_x, max_y = get_max_coord(group)
        min_x, min_y = get_min_coord(group)
        Image.fromarray(ori_img[min_y - padding:max_y + padding, min_x - padding : max_x + padding]).save(f"workdir/datasets/{img_name}/{i}.png")
        group_coords.append([min_x, min_y, max_x, max_y])
    json.dump(group_coords, open(f"workdir/datasets/coords_{img_name}.json", "w"))