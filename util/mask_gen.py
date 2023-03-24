"""
# -*- coding: utf-8 -*-
# @Time    : 2023/3/18
# @Author  : ThomasTse
# @File    : mask_gen.py
# @IDE     : PyCharm
"""

# Tools lib
import os
import numpy as np
import cv2


def get_binary_mask(image, back_gt):
    mean_image = np.mean(image, 2)
    mean_back_gt = np.mean(back_gt, 2)
    diff = np.abs(mean_image - mean_back_gt)
    diff[diff <= 30] = 0
    diff[diff > 30] = 1
    # diff = diff[np.newaxis, np.newaxis, :, :]
    return diff

if __name__ == '__main__':
    input_dir = '../data/train/data/'
    gt_dir = '../data/train/gt/'
    output_dir = '../data/train/mask/'

    input_list = os.listdir(input_dir)
    gt_list = os.listdir(gt_dir)

    for i in range(len(input_list)):

        img = cv2.imread(input_dir + input_list[i])
        gt = cv2.imread(gt_dir + gt_list[i])
        tmp = input_list[i].split('_')
        num = tmp[0]
        mask = get_binary_mask(img, gt) * 255
        cv2.imwrite(output_dir + f"{num}_mask.png", mask)