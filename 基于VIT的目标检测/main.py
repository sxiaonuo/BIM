import os

import cv2
from PIL import Image
import random
import numpy as np
from utils.detect_line import detect_lines
from utils.detect_connect import detect_connect, get_max_coord, get_min_coord
from utils.hot import hot_detect_line, hot_dectect_connect
from tqdm import tqdm

import torch

from VIT.predict import predict
from VIT.vit import ViT

import json

def calculate_iou(bbox1, bbox2, cls1, cls2):
    if cls1 != cls2:
        return 0
    """
    计算两个框的交并比（IoU）。

    参数：
    bbox1: 第一个框的坐标，格式为 (x0, y0, x1, y1)
    bbox2: 第二个框的坐标，格式为 (x0, y0, x1, y1)

    返回值：
    iou: 两个框的交并比
    """
    # 计算两个框的相交部分的坐标
    inter_x0 = max(bbox1[0], bbox2[0])
    inter_y0 = max(bbox1[1], bbox2[1])
    inter_x1 = min(bbox1[2], bbox2[2])
    inter_y1 = min(bbox1[3], bbox2[3])

    # 计算相交部分的面积
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)

    # 计算两个框的面积
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # 计算交并比
    iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-6)  # 加上一个小数防止除零错误
    on = inter_area / (bbox2_area + 1e-6)
    # print(on)

    return on

def nms(bboxes, classes, scores, threshold=0.3):
    """
    使用非极大值抑制（NMS）去除重叠的框。

    参数：
    bboxes: 框的坐标，每行一个框，格式为 (x0, y0, x1, y1)
    classes: 每个框对应的类别
    scores: 每个框的置信度分数
    threshold: 重叠度阈值，小于该阈值的框会被保留，默认为 0.5

    返回值：
    selected_indices: 选中的框的索引列表
    """
    # 按照置信度分数降序排序
    sorted_indices = np.argsort(scores)[::-1]

    selected_indices = []
    while len(sorted_indices) > 0:
        # 选择置信度最高的框
        best_index = sorted_indices[0]
        selected_indices.append(best_index)

        # 计算当前选中框与其余框的交并比
        ious = [calculate_iou(bboxes[best_index], bboxes[idx], classes[best_index], classes[idx]) for idx in sorted_indices[1:]]

        # 去除重叠度高于阈值的框
        indices_to_keep = np.where(np.array(ious) <= threshold)[0]
        sorted_indices = sorted_indices[indices_to_keep + 1]
    return selected_indices


if __name__ == '__main__':
    """超参数"""
    img_name = 4
    padding = 20

    """"""
    # 连通发现
    ori_img = cv2.imread(f'../static/img/{img_name}.png')
    groups = hot_dectect_connect(f'../static/img/{img_name}.png', None, None, 0, path='../static/record/')
    groups = [group for group in groups if len(group) >= 4]
    print(len(groups))

    # 检测
    group_patch = []
    for i, group in enumerate(tqdm(groups)):
        max_x, max_y = get_max_coord(group)
        min_x, min_y = get_min_coord(group)
        if max_y - max_x > 2000 or max_x - min_x > 2000:
            continue
        group_patch.append([min_x, min_y, max_x, max_y])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = ViT(in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=10).to(device)
    vit.load_state_dict(torch.load('../VIT/vit_model.pth'))

    res = []
    scores = []
    batch_size = 16
    for i in tqdm(range(0, len(group_patch), batch_size)):
        batch_group_patch = group_patch[i:i + batch_size]
        batch_img = []
        for it in batch_group_patch:
            img = ori_img[it[1] - padding:it[3] + padding, it[0] - padding:it[2] + padding]
            batch_img.append(img)

        a, score = predict(vit, batch_img, device)
        res.extend(a)
        scores.extend(score)
    print(len(group_patch), len(res), len(scores))
    target = [it for i, it in enumerate(group_patch) if res[i] == 1]
    scores = [it for i, it in enumerate(scores) if res[i] == 1]
    target = np.array(target)
    scores = np.array(scores)
    classes = np.ones_like(scores)
    keep = nms(target,classes, scores, 0.5)
    print(len(keep))


    for it in keep:
        cv2.rectangle(ori_img, (target[it][0], target[it][1]), (target[it][2], target[it][3]), (0, 255, 0), 2)

    os.makedirs("workdir/run", exist_ok=True)
    Image.fromarray(ori_img).save("workdir/run/0.png")




