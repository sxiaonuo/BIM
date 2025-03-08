import cv2
import numpy as np
import os
from skimage.morphology import skeletonize

def save_img(img,path):
    # 保存结果
    output_path = os.path.join(output_dir, path)
    cv2.imwrite(output_path, img)

def classify_contour(cnt, epsilon_factor=0.02):
    """分类轮廓为曲线或直线"""
    epsilon = epsilon_factor * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) > 2:
        return 'curve', cnt.squeeze(1)
    elif len(approx) == 2:
        # 提取直线端点
        start = tuple(approx[0][0])
        end = tuple(approx[1][0])
        return 'line', (start, end)
    return None, None


def extract_shapes(image_path, min_contour_length=20):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件：{image_path}")

    # 预处理流程
    ksize = (3,3)
    thresh = 250
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, ksize, 0)
    _, binary = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY_INV)

    # 骨架化处理
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255

    save_img(skeleton, str(ksize) + '+' + str(thresh) + '+' + '.png')
    # 轮廓检测与分类
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    curves = []
    lines = []
    for cnt in contours:
        if len(cnt) < min_contour_length:
            continue
        shape_type, data = classify_contour(cnt, epsilon_factor=0.02)

        if shape_type == 'curve':
            curves.append(data)
        elif shape_type == 'line':
            lines.append(data)

    return curves, lines, skeleton


def save_result_image(original, curves, lines, output_dir):
    """生成并保存结果图"""
    # 创建带标注的示意图
    annotated_img = original.copy()

    # 绘制曲线（绿色）
    for curve in curves:
        cv2.polylines(annotated_img, [curve], False, (0, 255, 0), 1)

    # 绘制直线（红色）并标注端点
    for i, (start, end) in enumerate(lines):
        # 绘制直线
        cv2.line(annotated_img, start, end, (0, 0, 255), 1)

        # # 标注端点坐标
        # cv2.putText(annotated_img, f"S({start[0]},{start[1]})",
        #             (start[0] + 5, start[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # cv2.putText(annotated_img, f"E({end[0]},{end[1]})",
        #             (end[0] + 5, end[1] + 20),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    save_img(annotated_img,'detect_result.png')
    print(f"结果图已保存至：{output_dir}")


if __name__ == "__main__":
    input_path = "workdir/run/091810/ori.png"
    output_dir = "workdir/run/output"
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 提取形状
        curves, lines, skeleton = extract_shapes(input_path)

        # 保存结果图
        original_img = cv2.imread(input_path)
        save_result_image(original_img, curves, lines, output_dir)

        # 控制台输出
        print(f"检测到 {len(curves)} 条曲线和 {len(lines)} 条直线")
        # print("\n=== 直线列表 ===")
        # for i, (start, end) in enumerate(lines):
        #     print(f"直线 {i + 1}:")
        #     print(f"  起点坐标 (x, y): {start}")
        #     print(f"  终点坐标 (x, y): {end}")
        #     print(f"  线段长度: {np.linalg.norm(np.array(start) - np.array(end)):.2f} 像素")
        #     print("-" * 30)

    except Exception as e:
        print(f"处理失败：{str(e)}")