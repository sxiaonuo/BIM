import os
import cv2
import numpy as np
import torch
from torch import nn, optim

from VIT.vit import ViT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(model, imgs, device):
    """
    :param model:
    :param imgs: [nd.array, nd.array...]
    :param device:
    :return: 返回为列表和置信度
    """
    model.eval()
    inputs = torch.stack([transform(img) for img in imgs]).to(device)
    outputs = model(inputs)
    return outputs.argmax(dim=1).tolist(), outputs.max(dim=1).values.tolist()

def predict_img(model, img_path, device):
    imgs = []
    for img in os.listdir(img_path):
        print(img)
        img = cv2.imread(os.path.join(img_path, img))
        imgs.append(img)
    return predict(model=model, imgs=imgs, device=device)



if __name__ == '__main__':
    # imgs = []
    # ori_img = cv2.imread(f'../static/img/4.png')
    # # 随机裁剪10张
    # for i in range(10):
    #     x = np.random.randint(0, ori_img.shape[1] - 224)
    #     y = np.random.randint(0, ori_img.shape[0] - 224)
    #     size = np.random.randint(224, 256)
    #     imgs.append(ori_img[y:y + size, x:x + size])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vit = ViT(in_channels=3, patch_size=16, emb_size=768, img_size=224, depth=12, n_classes=10).to(device)
    vit.load_state_dict(torch.load('../VIT/vit_model.pth'))
    print(predict_img(model=vit, img_path="data/-1", device=device))