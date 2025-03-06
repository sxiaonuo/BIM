import operator
from collections import defaultdict

import cv2
from PIL import Image, ImageDraw
import numpy as np
import random
import os
import math
from tqdm.contrib import itertools

from yacs.config import CfgNode as CN

__all_ = ['get_cfg_defaults']

_C = CN()
_C.GET_LINKS_EPS = 1  # 目前只能用1， 否则连通检测会出错   # 哎，倒也未必会出错   # 现在这个值用来做其他事情了，一定得是1
_C.MAX_CONTINUE = 10  # 某个地方用到的关键参数，可调

_C.ELINE = CN()
_C.ELINE.EPS = 3  # eps
_C.ELINE.ANGLE = 1  # angle

_C.DETECT = CN()
_C.DETECT.WIDTH = 2048  # 这个版本的方法时间复杂度与图片尺寸呈近似线性关系，其实不再需要用裁剪的方式加快检测速度，因此我使用了2048的裁剪尺寸，仅为了不让等待那么焦虑。
_C.DETECT.HEIGHT = 2048

_C.SAVE_DIR = ''


def get_cfg_defaults(merge_from=None):
    cfg = _C.clone()
    if merge_from is not None:
        cfg.merge_from_other_cfg(merge_from)
    return cfg


def get_thin(img, cfg):
    # 原图转细线
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    # 闭运算
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # thin
    thin = cv2.bitwise_not(closing - binary)
    return thin

# 获取轮廓
def get_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nowhite = np.where(binary != 255)
    nowhite = set(zip(*nowhite))

    white = np.ones_like(img, dtype=np.uint8 ) * 255
    for x, y in nowhite:
        for dx, dy in zip([0, 1, 0, -1], [1, 0, -1, 0]):
            if (x + dx, y + dy) not in nowhite:
                white[x, y] = img[x, y]
    return white

class ELine:
    def __init__(self, pt1, pt2, type, k1, k2, color, volume):
        self.pt1 = pt1
        self.pt2 = pt2
        self.type = type
        self.dir = np.zeros(4, dtype=np.int32)  # 1, -1, inf, 0   # 不需要考虑后两种方向
        self.k1 = k1 # 斜率的分子
        self.k2 = k2 # 斜率的分母
        self.color = color
        self.volume = volume

    def has_dir(self, dirs):
        """是否含有相同方向"""
        return sum(dirs & self.dir) > 0

    def dir_eq(self, dirs):
        return (dirs == self.dir).all()

    def append_dir(self, dir):
        assert dir in [1, -1, np.inf, 0]
        if dir == 1:
            self.dir[0] = 1
        elif dir == -1:
            self.dir[1] = 1
        elif dir == np.inf:
            self.dir[2] = 1
        else:
            self.dir[3] = 1


def get_eline(img, cfg):
    """
    目前主要就是这个函数慢，可以用numpy来优化，能大大加快

    :param img: ndarray 图片
    :param cfg:
    :return: 带有斜率，颜色的线元
    """
    thin = get_thin(img, cfg)
    w, h = thin.shape[:2]
    thin_pad = np.ones((w + 1, h + 1), dtype=np.uint8) * 255
    thin_pad[:-1, :-1] = thin
    img_pad = np.ones((w + 1, h + 1, 3), dtype=np.uint8) * 255
    img_pad[:-1, :-1, :] = img
    h_line = []
    for y in range(w + 1):
        tp = None
        for x in range(h + 1):
            if thin_pad[y, x] == 0:
                if tp is None:
                    tp = x
                elif (img[y, x] != img[y, tp]).any():
                    h_line.append([y, tp, y, x - 1])
                    tp = x
            else:
                if tp is not None:
                    h_line.append([y, tp, y, x - 1])
                    tp = None

    v_line = []
    for x in range(h + 1):
        tp = None
        for y in range(w + 1):
            if thin_pad[y, x] == 0:
                if tp is None:
                    tp = y
                elif (img[y, x] != img[tp, x]).any():
                    v_line.append([tp, x, y - 1, x])
                    tp = y
            else:
                if tp is not None:
                    v_line.append([tp, x, y - 1, x])
                    tp = None

    dot1 = [(line[0], line[1]) for line in h_line if (line[0], line[1]) == (line[2], line[3])]
    dot2 = [(line[0], line[1]) for line in v_line if (line[0], line[1]) == (line[2], line[3])]
    dots = list(set(dot2) - (set(dot2) - set(dot1)))
    dots = [[d[0], d[1], d[0], d[1]] for d in dots]
    h_line = [line for line in h_line if (line[0], line[1]) != (line[2], line[3])]
    v_line = [line for line in v_line if (line[0], line[1]) != (line[2], line[3])]
    # 创建Eline, 上颜色，上方向
    elines = []
    for eline in [*v_line, *h_line, *dots]:
        pt1 = (eline[0], eline[1])
        pt2 = (eline[2], eline[3])
        color = tuple([int(it) for it in img[pt1]])
        k = (pt2[1] - pt1[1] + 1) / (pt2[0] - pt1[0] + 1)
        volume = max(abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])) + 1

        el = ELine(pt1, pt2, 'dot' if pt1 == pt2 else 'line', abs(k), color, volume)
        # 加方向
        minx, maxx = sorted([pt1[0], pt2[0]])
        miny, maxy = sorted([pt1[1], pt2[1]])
        if tuple(img_pad[minx - 1, miny - 1]) == color:
            el.append_dir(1)
        if tuple(img_pad[maxx + 1, maxy + 1]) == color:
            el.append_dir(1)
        if tuple(img_pad[maxx + 1, miny - 1]) == color:
            el.append_dir(-1)
        if tuple(img_pad[minx - 1, maxy + 1]) == color:
            el.append_dir(-1)

        elines.append(el)

    return elines


def get_eline_faster(img, cfg):
    """
    快的get_eline

    :param img: ndarray 图片
    :param cfg:
    :return: 带有斜率，颜色的线元
    """
    thin = get_thin(img, cfg)
    w, h = thin.shape[:2]
    thin_pad = np.ones((w + 1, h + 1), dtype=np.uint8) * 255
    thin_pad[:-1, :-1] = thin
    img_pad = np.ones((w + 1, h + 1, 3), dtype=np.uint8) * 255
    img_pad[:-1, :-1, :] = img
    h_line = []

    nowhite = np.where(thin_pad != 255)
    nowhite = list(zip(*nowhite))
    if len(nowhite) == 0:
        return []
    n1 = sorted(nowhite, key=lambda x: (x[0], x[1]))
    n2 = sorted(nowhite, key=lambda x: (x[1], x[0]))
    head = n1[0]
    tail = n1[0]
    for i, it in enumerate(n1[1:]):
        if it[0] == n1[i][0] and it[1] - n1[i][1] == 1 and (img[it] == img[n1[i]]).all():
            tail = it
            continue
        h_line.append([head[0], head[1], tail[0], tail[1]])
        head = it
        tail = it
    if head != tail:
        h_line.append([head[0], head[1], tail[0], tail[1]])

    v_line = []
    for i, it in enumerate(n2[1:]):
        if it[1] == n2[i][1] and it[0] - n2[i][0] == 1 and (img[it] == img[n2[i]]).all():
            tail = it
            continue
        v_line.append([head[0], head[1], tail[0], tail[1]])
        head = it
        tail = it
    if head != tail:
        v_line.append([head[0], head[1], tail[0], tail[1]])

    dot1 = [(line[0], line[1]) for line in h_line if (line[0], line[1]) == (line[2], line[3])]
    dot2 = [(line[0], line[1]) for line in v_line if (line[0], line[1]) == (line[2], line[3])]
    dots = list(set(dot2) - (set(dot2) - set(dot1)))
    dots = [[d[0], d[1], d[0], d[1]] for d in dots]
    h_line = [line for line in h_line if (line[0], line[1]) != (line[2], line[3])]
    v_line = [line for line in v_line if (line[0], line[1]) != (line[2], line[3])]
    # 创建Eline, 上颜色，上方向
    elines = []
    for eline in [*v_line, *h_line, *dots]:
        pt1 = (eline[0], eline[1])
        pt2 = (eline[2], eline[3])

        # 生成所有xy坐标
        x_coords = range(pt1[0], pt2[0] + 1) if pt2[0] > pt1[0] else range(pt1[0], pt2[0] - 1, -1)
        y_coords = range(pt1[1], pt2[1] + 1) if pt2[1] > pt1[1] else range(pt1[1], pt2[1] - 1, -1)
        colors = defaultdict(int)
        for x, y in zip(x_coords, y_coords):
            colors[tuple([int(it) for it in img[(x, y)]])] += 1
        color = max(colors, key=colors.get)

        # color = tuple([int(it) for it in img[pt1]])

        k1 = pt2[1] - pt1[1]
        k2 = pt2[0] - pt1[0]
        volume = max(abs(pt2[0] - pt1[0]), abs(pt2[1] - pt1[1])) + 1

        ### 补丁，在这里吧eline的xy坐标转回去
        el = ELine((pt1[1], pt1[0]), (pt2[1], pt2[0]), 'dot' if pt1 == pt2 else 'line', k1, k2, color, volume)
        # 加方向
        minx, maxx = sorted([pt1[0], pt2[0]])
        miny, maxy = sorted([pt1[1], pt2[1]])
        if tuple(img_pad[minx - 1, miny - 1]) == color:
            el.append_dir(1)
        if tuple(img_pad[maxx + 1, maxy + 1]) == color:
            el.append_dir(1)
        if tuple(img_pad[maxx + 1, miny - 1]) == color:
            el.append_dir(-1)
        if tuple(img_pad[minx - 1, maxy + 1]) == color:
            el.append_dir(-1)

        elines.append(el)

    return elines


class ChainForwardStar:
    # 链式前向星
    def __init__(self, num_node, num_edge):
        self.num_node = num_node
        self.h = np.zeros(num_node, dtype=np.int32)
        self.e = np.zeros(num_edge, dtype=np.int32)
        self.ne = np.zeros(num_edge, dtype=np.int32)
        self.idx = 0
        for i in range(num_node):
            self.h[i] = -1

    def add_edge(self, u, v):
        self.e[self.idx] = v
        self.ne[self.idx] = self.h[u]
        self.h[u] = self.idx
        self.idx += 1

    def get_neighbors(self, u):
        neighbors = []
        p = self.h[u]
        while p != -1:
            neighbors.append(self.e[p])
            p = self.ne[p]
        return neighbors


class UnionFind:
    # 并查集
    def __init__(self, size):
        # 初始化每个元素的父节点为自己
        self.parent = list(range(size))
        # 可选：初始化每个集合的元素数量，如果不需要统计集合大小可以省略
        self.rank = [1] * size

    def find(self, x):
        """查找x的根节点，使用路径压缩"""
        if self.parent[x] != x:
            # 路径压缩：将x的父节点设为其根节点的父节点
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并x和y所在的集合"""
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # 如果x和y不在同一个集合中，则将较小的集合合并到较大的集合上
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                self.rank[rootX] += self.rank[rootY]
            else:
                self.parent[rootX] = rootY
                self.rank[rootY] += self.rank[rootX]

    def connected(self, x, y):
        """判断x和y是否在同一集合中"""
        return self.find(x) == self.find(y)

    def get_sets(self):
        """返回并查集中所有连通集合的列表"""
        # 创建一个字典，键是根节点，值是属于该根节点的集合元素列表
        sets_dict = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in sets_dict:
                sets_dict[root] = []
            sets_dict[root].append(i)

        # 将字典转换为列表
        return list(sets_dict.values())


def getlinks(elines, cfg):
    # 预处理线元的相邻关系
    eps = cfg.GET_LINKS_EPS

    num_node = len(elines)
    num_edge = num_node * (2 * eps + 1) ** 2  # 你猜这是啥意思
    cfs = ChainForwardStar(num_node, num_edge)
    # 反向处理，减时间复杂度
    dic = {}
    for i, eline in enumerate(elines):
        dic[eline.pt1] = i
        dic[eline.pt2] = i
    # 链式前向星存图
    for i, eline in enumerate(elines):
        for pt in [eline.pt1, eline.pt2]:
            # for r in range(1, eps + 1):
            r = eps
            for dx in range(pt[0] - r, pt[0] + r + 1):
                for dy in range(pt[1] - r, pt[1] + r + 1):
                    if (dx, dy) in dic and i != dic[(dx, dy)]:
                        cfs.add_edge(i, dic[(dx, dy)])
    return cfs

def merge_eline(group):
    dots = []
    color = group[0].color
    for eline in group:
        dots.append(eline.pt1)
        dots.append(eline.pt2)
    dots = sorted(dots, key=lambda x: x[0])
    x0 = dots[0][0]
    x1 = dots[-1][0]
    y0 = max(row[1] for row in dots)
    y1 = min(row[1] for row in dots)
    if not(((x0,y0) in dots) and ((x1,y1) in dots)):
        y0,y1 = y1,y0
    return [(x0, y0), (x1, y1), color]

def merge_line(group):
    # 黏合线
    dots = []
    color = group[0][2]
    for line in group:
        dots.append(line[0])
        dots.append(line[1])
    dots = sorted(dots, key=lambda x: x[0])
    # dots = sorted(dots, key=operator.itemgetter(0,1))
    x0 = dots[0][0]
    y0 = dots[0][1]
    x1 = dots[-1][0]
    y1 = dots[-1][1]
    return [(x0, y0), (x1, y1), color]


def detect_one_img(img, cfg):
    """ 返回的是一堆合并完的线，而且上了颜色，线的性质暂时不是非常清楚 """
    # elines 返回值:带有斜率，颜色的线元
    elines = get_eline_faster(img, cfg)
    cfs = getlinks(elines, cfg)
    elines_group_by_line = []

    # 先遍历1-5纯血直线
    group_by_volume = {}
    for idx, eline in enumerate(elines):
        if eline.volume not in group_by_volume:
            group_by_volume[eline.volume] = []
        group_by_volume[eline.volume].append(idx)
    uf = UnionFind(len(elines))
    for it in range(1, 6):
        if it not in group_by_volume:
            continue
        for eline_idx in group_by_volume[it]:
            eline = elines[eline_idx]
            for eline2_idx in cfs.get_neighbors(eline_idx):
                eline2 = elines[eline2_idx]
                if (eline.color == eline2.color and eline.volume == eline2.volume and eline.dir_eq(eline2.dir)
                        and (eline.volume == 1 or (eline.k1 + eline2.k2) and (eline2.k1 + eline.k2))):
                    uf.union(eline_idx, eline2_idx)

    sets = uf.get_sets()
    # 剩余的线  用来再做一次上面的操作
    other_elines = []
    for i, set_ in enumerate(sets):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if len(set_) >= cfg.MAX_CONTINUE:
            elines_group_by_line.append(set_)
        else:
            for j in set_:
                other_elines.append(j)

    # 混合遍历 , 大1 小1 等于
    group_by_volume = {}
    for idx in other_elines:
        eline = elines[idx]
        if eline.volume not in group_by_volume:
            group_by_volume[eline.volume] = []
        group_by_volume[eline.volume].append(idx)

    uf = UnionFind(len(elines))  # 重置并查集
    for key in group_by_volume.keys():
        for di in [-1, 0, 1]:
            if key + di in group_by_volume.keys():
                for eline1_idx in group_by_volume[key]:
                    eline1 = elines[eline1_idx]
                    for eline2_idx in cfs.get_neighbors(eline1_idx):
                        eline2 = elines[eline2_idx]
                        if eline1.color == eline2.color and eline2_idx in group_by_volume[key + di] and eline1.dir_eq(
                                eline2.dir) and (
                                eline1.volume == 1 or (eline1.k1 + eline2.k2) and (eline2.k1 + eline1.k2)):
                            uf.union(eline1_idx, eline2_idx)
    # 合并罢了
    sets = uf.get_sets()
    for i, set_ in enumerate(sets):
        if len(set_) > 1:
            elines_group_by_line.append(set_)
        else:
            if set_[0] in other_elines:
                elines_group_by_line.append(set_)

    len(elines_group_by_line)
    egbl = []  # 真的不知道该怎么起名了
    for group in elines_group_by_line:
        egbl.append([elines[i] for i in group])
    # 黏合线元为直线，包含上颜色
    lines = [merge_eline(group) for group in egbl]

    return lines, elines



def _get_k(line):
    x0, y0, x1, y1 = line
    if x1 - x0 == 0:
        return float('inf')
    return (y1 - y0) / (x1 - x0)


def _get_b(line):
    x0, y0, x1, y1 = line
    if x1 - x0 == 0:
        return float('inf')
    return y0 - _get_k(line) * x0


def _angle_between_lines(line1, line2):
    # 计算角度
    m1 = _get_k(line1)
    m2 = _get_k(line2)

    if m1 == float('inf'):
        m1 = 99999999999
    if m2 == float('inf'):
        m2 = 99999999999
    if m1 * m2 == -1:
        return 90.0  # 直线垂直
    tan_theta = abs((m1 - m2) / (1 + m1 * m2))
    theta_radians = math.atan(tan_theta)
    theta_degrees = math.degrees(theta_radians)

    return theta_degrees


def _is_overlap(line1, line2):
    "判重，x轴无交集，y轴无交集"
    x0, y0, x1, y1 = line1
    x2, y2, x3, y3 = line2
    x0, x1 = sorted([x0, x1])
    x2, x3 = sorted([x2, x3])
    y0, y1 = sorted([y0, y1])
    y2, y3 = sorted([y2, y3])
    if x0 > x3 or x1 < x2:
        return False
    if y0 > y3 or y1 < y2:
        return False
    return True


def _length(line):
    #
    x0, y0, x1, y1 = line
    return math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)


def concat_line(total_lines, cfg):
    # 过滤很短的线
    total_lines = [line for line in total_lines if _length([*line[0], *line[1]]) > 3]

    # 反向预处理
    eps = cfg.ELINE.EPS

    num_node = len(total_lines)
    num_edge = num_node * (2 * eps + 1) ** 2  # 你猜这是啥意思
    cfs = ChainForwardStar(num_node, num_edge)
    # 反向处理，减时间复杂度
    dic = {}
    for i, line in enumerate(total_lines):
        dic[line[0]] = i
        dic[line[1]] = i
    # 链式前向星存图
    for i, line in enumerate(total_lines):
        for pt in [line[0], line[1]]:
            for r in range(1, eps + 1):
                for dx in range(pt[0] - r, pt[0] + r + 1):
                    for dy in range(pt[1] - r, pt[1] + r + 1):
                        if (dx, dy) in dic and i != dic[(dx, dy)]:
                            cfs.add_edge(i, dic[(dx, dy)])
    # 将应该合并的线合并
    uf = UnionFind(len(total_lines))
    for line1_idx, line in enumerate(total_lines):
        color1 = line[2]
        line1 = [*line[0], *line[1]]
        for line2_idx in cfs.get_neighbors(line1_idx):
            line2 = total_lines[line2_idx]
            color2 = line2[2]
            line2 = [*line2[0], *line2[1]]
            # 颜色相同，角度小于阈值，且不"重叠"
            if color1 == color2 and _angle_between_lines(line1, line2) < cfg.ELINE.ANGLE and not _is_overlap(line1,line2):
                # 补丁
                if _get_k(line1) == float('inf') and line1[0] != line2[0]:
                    continue
                elif _get_k(line1) == 0 and line1[1] != line2[1]:
                    continue
                # 连
                uf.union(line1_idx, line2_idx)
    # 合并罢了
    sets = uf.get_sets()
    uni_lines = []
    for s in sets:
        t = []
        for i in s:
            t.append(total_lines[i])
        uni_lines.append(t)
    uni_lines = [merge_line(uni_line) for uni_line in uni_lines]
    return uni_lines


def detect_lines(img, cfg=get_cfg_defaults()):
    # 裁剪。这个版本的方法时间复杂度与图片尺寸呈近似线性关系，其实不再需要用裁剪的方式加快检测速度，因此我使用了2048的裁剪尺寸，仅为了不让等待那么焦虑。
    if img.shape[:2] < (cfg.DETECT.HEIGHT, cfg.DETECT.WIDTH):
        lines, all_elines = detect_one_img(img, cfg)
        all_elines = [[eline.pt1, eline.pt2, eline.color] for eline in all_elines]
        return concat_line(lines, cfg), lines, all_elines

    # 计算裁剪坐标
    img_width, img_height = img.shape[:2]
    crop_width, crop_height = cfg.DETECT.WIDTH, cfg.DETECT.HEIGHT
    step_heights = [i for i in range(0, img_height, crop_height)]
    step_widths = [i for i in range(0, img_width, crop_width)]

    itr = itertools.product(step_widths, step_heights)

    # 裁剪，检测每一个部分
    total_lines = []
    all_elines = []
    for i, j in itr:
        crop_img = img[i:min(i + crop_width, img_width), j:min(j + crop_height, img_height)]
        lines, elines = detect_one_img(crop_img, cfg)
        total_lines.extend(
            [[(line[0][0] + j, line[0][1] + i), (line[1][0] + j, line[1][1] + i), line[2]] for line in lines])
        all_elines.extend(
            [[(eline.pt1[0] + j, eline.pt1[1] + i), (eline.pt2[0] + j, eline.pt2[1] + i), eline.color] for eline in elines]
        )
    # 合并不同patch的直线
    lines = concat_line(total_lines, cfg)

    return lines, total_lines, all_elines


if __name__ == '__main__':
    # 读取配置
    cfg = get_cfg_defaults()
    cfg.SAVE_DIR = f'workdir/run/{random.randint(0, 100000):06d}/'
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(cfg)

    #######################################################
    # 示例代码
    # ori_img = cv2.imread('../src/1.png')
    # ori_img = ori_img[3000:-3000, 3000:-3000, :]  # 为了更快看到结果，只截取一部分
    ori_img = cv2.imread('../static/img/b1.png')
    ori_img = ori_img[5000:8000, 5000:7000, :]
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(ori_img).save(cfg.SAVE_DIR + 'ori.png')
    lines, total_lines, all_elines = detect_lines(ori_img, cfg)
    ######################################################
    # 绘制
    print('num of lines', len(lines))
    for idx, line in enumerate(lines):
        print(line)
        if idx == 5:
            print("...")
            break
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

    white = np.ones_like(ori_img) * 255
    for line in total_lines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(white, line[0], line[1], color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'nocancat_randomcolor.png')
    white = np.ones_like(ori_img) * 255
    for line in total_lines:
        cv2.line(white, line[0], line[1], line[2], 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'nocancat_originalcolor.png')

    white = np.ones_like(ori_img) * 255
    for line in all_elines:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(white, line[0], line[1], color, 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'eline_randomcolor.png')
    white = np.ones_like(ori_img) * 255
    for line in all_elines:
        cv2.line(white, line[0], line[1], line[2], 1)
    Image.fromarray(white).save(cfg.SAVE_DIR + 'eline_originalcolor.png')