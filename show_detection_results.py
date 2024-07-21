import os
import cv2
import numpy as np


from PIL import Image
# 将分割预测的图片添加颜色
colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]

def colorize_mask(mask, orininal_h, orininal_w):
    seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(mask, [-1])], [orininal_h, orininal_w, -1])

    image   = Image.fromarray(np.uint8(seg_img))
    return image

mask_path = '/root/spectral-segmentation/output/2024_03_26_06_18_02_newafft_fusion_rec/detection-results'
save_path = '/root/spectral-segmentation/output/2024_03_26_06_18_02_newafft_fusion_rec/detection-results-color'
mask_list = os.listdir(mask_path)
for mask in mask_list:
    mask_img = cv2.imread(os.path.join(mask_path, mask), cv2.IMREAD_GRAYSCALE)
    mask_img = colorize_mask(mask_img, mask_img.shape[0], mask_img.shape[1])
    mask_img.save(os.path.join(save_path, mask))