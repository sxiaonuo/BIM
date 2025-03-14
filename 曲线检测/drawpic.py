import random

import cv2
import numpy as np
from PIL import Image


def draw_line_r(ori_img,lines,name,cfg):
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pt1 = line[0]
        pt2 = line[1]
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'randomcolor' + name + '.png')
    return white

def draw_line_o(ori_img, lines, name, cfg):
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = line[2]
        pt1 = line[0]
        pt2 = line[1]
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'originalcolor' + name + '.png')
    return white

def draw_Eline_r(ori_img, lines, name, cfg):
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        pt1 = line.pt1
        pt2 = line.pt2
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'randomcolor' + name + '.png')
    return white

def draw_Eline_o(ori_img, lines, name, cfg):
    white = np.ones_like(ori_img) * 255
    for line in lines:
        color = line.color
        pt1 = line.pt1
        pt2 = line.pt2
        cv2.line(white, pt1, pt2, color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'originalcolor' + name + '.png')
    return white
