import os
import random

import cv2
import numpy as np

import random
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import distance
import tqdm
from PIL import Image


def color_stratify(img_fn, save_dir, num_colors):
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    nowhite = img[binary == 0]
    print("彩色像素占比: {:.3f}% ".format(len(nowhite) * 100 / (img.shape[0] * img.shape[1])))
    nowhite_list = nowhite.tolist()
    random.shuffle(nowhite_list)
    data = np.array(nowhite_list[:10000])

    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(data)
    center = kmeans.cluster_centers_.astype(np.int32)

    imgs = np.ones((num_colors, img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

    nowhite_index = np.where(binary == 0)
    nowhite_index = list(zip(*nowhite_index))
    for i, j in tqdm.tqdm(nowhite_index):
        pixel = img[i, j]
        # 计算像素与所有中心颜色的距离
        distances = [distance.euclidean(pixel, center_color) for center_color in center]
        # 找到距离最近的中心颜色的索引
        closest_center_idx = np.argmin(distances)
        # 将像素值赋给对应位置
        imgs[closest_center_idx, i, j] = pixel

    for idx, img in enumerate(imgs):
        Image.fromarray(img).save(save_dir + 'color_' + str(idx) + '.png')

def optimized_color_stratify(
    img_fn: str,
    save_dir: str,
    num_colors: int,
    max_samples: int = 10000,
    batch_size: int = 1000
) -> None:
    """
    优化后的颜色分层函数

    参数:
        img_fn: 输入图像路径
        save_dir: 结果保存目录
        num_colors: 颜色分层数量
        max_samples: KMeans最大采样数 (默认10000)
        batch_size: 处理像素的批大小 (默认1000)
    """
    os.makedirs(save_dir, exist_ok=True)

    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 获取非白色像素 (使用numpy布尔索引)
    nowhite_mask = binary == 0
    nowhite_pixels = img[nowhite_mask]
    pixel_ratio = np.sum(nowhite_mask) * 100 / (img.shape[0] * img.shape[1])
    print(f"彩色像素占比: {pixel_ratio:.3f}%")

    # 采样策略优化 - 避免全量数据处理
    if len(nowhite_pixels) > max_samples:
        indices = np.random.choice(len(nowhite_pixels), max_samples, replace=False)
        sample_data = nowhite_pixels[indices]
    else:
        sample_data = nowhite_pixels

    # 使用更快的MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_colors,
        batch_size=min(256, len(sample_data)),  # 小批量大小
        max_iter=50,
        n_init=3,
        random_state=42
    )
    kmeans.fit(sample_data)
    centers = kmeans.cluster_centers_.astype(np.int32)

    # 预计算所有距离 (向量化操作)
    # 分割大数组以避免内存问题
    num_batches = int(np.ceil(len(nowhite_pixels) / batch_size))
    labels = np.zeros(len(nowhite_pixels), dtype=np.int32)

    for i in tqdm(range(num_batches), desc="处理像素批次"):
        start = i * batch_size
        end = start + batch_size
        batch = nowhite_pixels[start:end]

        # 向量化距离计算
        distances = np.linalg.norm(
            batch[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2
        )
        labels[start:end] = np.argmin(distances, axis=1)

    imgs = np.full((num_colors, *img.shape), 255, dtype=np.uint8)

    coords = np.argwhere(nowhite_mask)

    for color_idx in range(num_colors):
        mask = labels == color_idx
        if np.any(mask):
            imgs[color_idx, coords[mask, 0], coords[mask, 1]] = nowhite_pixels[mask]

    for idx in range(num_colors):
        Image.fromarray(imgs[idx]).save(os.path.join(save_dir, f'color_{idx}.png'))

import torch
import torch.nn.functional as F

def color_stratify_gpu(
    img_fn: str,
    save_dir: str,
    num_colors: int,
    use_gpu: bool = None,  # None=自动选择，True=强制GPU，False=强制CPU
    max_samples: int = 10000,
    batch_size: int = 4096,
    kmeans_iter: int = 10
) -> None:
    """
    终极优化版（自动GPU/CPU切换）

    参数:
        use_gpu: None=自动, True=强制GPU, False=强制CPU
        batch_size: GPU内存优化参数（越大越快，但需要更多显存）
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 高效读取图像（保持CPU端）
    img = cv2.imread(img_fn)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 2. 快速二值化（CPU）
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # 3. 设备选择逻辑
    if use_gpu is None:
        use_gpu = torch.cuda.is_available() and (h * w > 200000)  # 仅对>200KPix图像启用GPU

    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f"运行设备: {device} | 采样数: {max_samples} | 批大小: {batch_size}")

    # 4. 获取非白色像素（CPU端预处理）
    nowhite_mask = binary == 0
    nowhite_pixels = img[nowhite_mask]
    coords = np.argwhere(nowhite_mask)

    # 5. 采样策略（避免全量数据处理）
    if len(nowhite_pixels) > max_samples:
        indices = np.random.choice(len(nowhite_pixels), max_samples, replace=False)
        sample_data = nowhite_pixels[indices]
    else:
        sample_data = nowhite_pixels

    # 6. 转换数据到PyTorch
    sample_tensor = torch.from_numpy(sample_data).float().to(device)

    # 7. GPU/CPU通用KMeans实现
    centers = _accelerated_kmeans(
        sample_tensor, num_colors,
        max_iter=kmeans_iter,
        device=device
    )

    # 8. 分批次处理所有像素（避免内存溢出）
    num_batches = int(np.ceil(len(nowhite_pixels) / batch_size))
    all_labels = np.empty(len(nowhite_pixels), dtype=np.int32)

    for i in tqdm(range(num_batches), desc="处理像素"):
        start = i * batch_size
        end = start + batch_size
        batch_pixels = nowhite_pixels[start:end]

        # GPU/CPU统一处理
        batch_tensor = torch.from_numpy(batch_pixels).float().to(device)
        distances = torch.cdist(batch_tensor, centers.float())
        all_labels[start:end] = torch.argmin(distances, dim=1).cpu().numpy()

    # 9. 生成结果图像（使用numpy高级索引）
    imgs = np.full((num_colors, h, w, 3), 255, dtype=np.uint8)
    for color_idx in range(num_colors):
        mask = all_labels == color_idx
        imgs[color_idx, coords[mask, 0], coords[mask, 1]] = nowhite_pixels[mask]

    # 10. 保存结果
    for idx in range(num_colors):
        Image.fromarray(imgs[idx]).save(os.path.join(save_dir, f'color_{idx}.png'))

def _accelerated_kmeans(
    data: torch.Tensor,
    num_clusters: int,
    max_iter: int = 10,
    device: torch.device = None
) -> torch.Tensor:
    """优化的KMeans实现（支持GPU）"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机初始化中心点
    centers = data[torch.randperm(len(data))[:num_clusters]].clone()

    for _ in range(max_iter):
        # 计算距离并分配标签（完全向量化）
        distances = torch.cdist(data, centers)
        labels = torch.argmin(distances, dim=1)

        # 更新中心点（避免循环）
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(num_clusters, device=device)

        for i in range(num_clusters):
            mask = labels == i
            if mask.any():
                new_centers[i] = data[mask].mean(dim=0)
                counts[i] = mask.sum()

        # 处理空簇
        empty_mask = counts == 0
        if empty_mask.any():
            new_centers[empty_mask] = data[torch.randperm(len(data))[:empty_mask.sum()]]

        # 提前终止检查
        if torch.allclose(centers, new_centers, atol=1e-4):
            break

        centers = new_centers

    return centers.round().int()