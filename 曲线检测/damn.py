import random

import cv2
import numpy as np
import os
from skimage.morphology import skeletonize


def is_curve(contour, epsilon_factor=0.02):
    """使用多边形近似判断是否为曲线"""
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return len(approx) > 2  # 顶点数大于2视为曲线


def extract_curves(image_path):
    # 读取图像并校验
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件：{image_path}")

    # 预处理流程
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV)

    # 骨架化处理
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    # 轮廓检测与过滤
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    curves = []
    for cnt in contours:
        if len(cnt) > 20 and is_curve(cnt):  # 双重过滤条件
            points = cnt.squeeze(1)
            curves.append(points)

    return curves, skeleton


def save_curve_image(original, curves, output_path):
    """生成并保存曲线结果图"""
    result_img = original.copy()
    thicktwo = original.copy()
    thickone = original.copy()

    # 修正颜色生成语句
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in curves]

    # 绘制曲线
    for curve, color in zip(curves, colors):
        cv2.polylines(thicktwo, [curve], False, color, 2)
        cv2.polylines(thickone, [curve], False, color, 1)
        cv2.polylines(result_img, [curve], False, (255,255,255), 1)

    # 保存结果
    cv2.imwrite(output_path+'/result_img.png', result_img)
    cv2.imwrite(output_path+'/thicktwo.png', thicktwo)
    cv2.imwrite(output_path+'/thickone.png', thickone)
    print(f"结果已保存至：{output_path}")


if __name__ == "__main__":
    # 输入输出路径配置
    input_path = "workdir/run/091810/ori.png"
    output_dir = f'workdir/run/{random.randint(0, 100000):06d}/'
    # output_filename = "curve_result.png"

    try:
        # 自动创建目录
        os.makedirs(output_dir, exist_ok=True)

        # 处理图像
        curves, skeleton = extract_curves(input_path)

        # 保存结果
        original_img = cv2.imread(input_path)
        output_path = os.path.join(output_dir)
        save_curve_image(original_img, curves, output_path)

        # 控制台输出
        print(f"检测到 {len(curves)} 条曲线")
        for i, curve in enumerate(curves):
            print(f"曲线 {i + 1}: 包含 {len(curve)} 个点")

    except Exception as e:
        print(f"处理失败：{str(e)}")


