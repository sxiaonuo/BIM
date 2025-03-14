from utils.detect_line import *
from PIL import Image

from drawpic import *

def remove_bigelines(ori_img,vol_threshold):
    # new_img是消除线元后的原图
    new3_img = ori_img.copy()
    for eline in elines:
        if eline.volume >= vol_threshold:
            new3_img = cv2.line(new3_img,eline.pt1,eline.pt2,(255,255,255),1)
    return new3_img

if __name__ == '__main__':
    # 读取配置
    cfg = get_cfg_defaults()
    cfg.SAVE_DIR = f'workdir/run/{random.randint(0, 100000):06d}/'
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    print(cfg)

    ori_img = cv2.imread("../src/1.png")
    ori_img = ori_img[3000:-3000, 3000:-3000, :]  # 为了更快看到结果，只截取一部分
    Image.fromarray(ori_img).save(cfg.SAVE_DIR + 'ori.png')

    elines = get_eline_faster(ori_img,cfg)

    elines_img = draw_Eline_r(ori_img,elines,'elines',cfg)

    Image.fromarray(remove_bigelines(ori_img,3)).save(cfg.SAVE_DIR + 'new3.png')
    Image.fromarray(remove_bigelines(ori_img,4)).save(cfg.SAVE_DIR + 'new4.png')
    Image.fromarray(remove_bigelines(ori_img,5)).save(cfg.SAVE_DIR + 'new5.png')
    Image.fromarray(remove_bigelines(ori_img,6)).save(cfg.SAVE_DIR + 'new6.png')
