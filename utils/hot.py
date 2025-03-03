import json
import os

import cv2
from PIL import Image
import random
import numpy as np
from utils.detect_line import detect_lines

def hot_detect_line(fp, width=None, height=None, cfg=None, path='../static/record', reload=False):
    # 计算 key
    fn = os.path.basename(fp)
    key = fn + f"{width, height}"
    # 打开reocrd
    if not os.path.exists(os.path.join(path, "record.json")):
        os.makedirs(path, exist_ok=True)
        json.dump({}, open(os.path.join(path, "record.json"), "w"))
    with open(os.path.join(path, "record.json")) as record_f:
        record = json.load(record_f)
    # 如果没有记录则reload
    if key not in record:
        reload = True
    if reload:
        ori_img = cv2.imread(fp)
        if width is None:
            width = [0, ori_img.shape[0]]
        if height is None:
            height = [0, ori_img.shape[1]]
        ori_img = ori_img[width[0]:width[1], height[0]:height[1]]
        lines, _, _ = detect_lines(ori_img)
        lines = [[(int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (int(line[2][0]), int(line[2][1]), int(line[2][2]))] for line in lines] # 补丁
        save_fp = os.path.join(path, f"{random.randint(0, 100000)}.json")
        with open(save_fp, "w") as f:
            f.write(json.dumps(lines))
        record[key] = save_fp
        with open(os.path.join(path, "record.json"), "w") as record_f:
            json.dump(record, record_f)
        return lines
    else:
        if width is None:
            width = [None, None]
        if height is None:
            height = [None, None]
        print(f"正在重载{fn}[{width[0]}:{width[1]}, {height[0]}:{height[1]}]")
        lines = json.load(open(record[key]))
        return lines


