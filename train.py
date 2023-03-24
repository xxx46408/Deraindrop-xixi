"""
# -*- coding: utf-8 -*-
# @Time    : 2023/3/18
# @Author  : ThomasTse
# @File    : train.py
# @IDE     : PyCharm
"""

# PyTorch lib
import torch
from torch.autograd import Variable

# Tools lib
import argparse
import os
import cv2
import sys
import numpy as np
import time
import math
import logging
import datetime
from tqdm import tqdm
from pathlib import Path
from util.logger import Logger

# Models lib
from models import *

# Metrics Lib
from util.metrics import calc_psnr, calc_ssim

# torch.set_default_tensor_type(torch.FloatTensor)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./data/train/data/', type=str, help='path of degraded raindrop images')
    parser.add_argument("--gt_dir", default='./data/train/gt/', type=str, help='path of clean images')
    parser.add_argument("--mask_dir", default='./data/train/mask/', type=str, help='path of mask data')
    parser.add_argument("--test_input_dir", default='./data/test_a/data/', type=str, help='path of test degraded raindrop images')
    parser.add_argument("--test_gt_dir", default='./data/test_a/gt/', type=str, help='path of test clean images')
    parser.add_argument("--gpus", default='0', type=str, help='index of gpu device:0, 1, 2, 3 or cpu')
    parser.add_argument("--model_weights", default='./weights/vgg16-397923af.pth', type=str, help='path of vgg model weight')
    parser.add_argument("--snapshot", default='./snapshot/', type=str, help='path to save checkpoint')
    # PARAMETER
    parser.add_argument("--theta", default=0.8, type=float, help='formula4: theta')
    parser.add_argument("--lr", default=0.0005, type=float, help='learning rate')
    parser.add_argument("--epoch", default=400, help='training epochs')
    parser.add_argument("--pre_epoch", default=0, help='training epochs')

    args = parser.parse_args()
    return args

# date format
def date_modified(path=__file__):
    # return human-readable file modification date, i.e. '2021-03-26'
    t = datetime.datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f'{t.year}-{t.month:02d}-{t.day:02d} ({t.hour:02d}.{t.minute:02d}.{t.second:02d})'


# logger
LOG_FORMAT = f'{date_modified()} - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# dataset
def align_to_four(img):
    a_row = int(img.shape[0] / 4) * 4
    a_col = int(img.shape[1] / 4) * 4
    img = img[0:a_row, 0:a_col]
    return img

def prepare_img_to_tensor(img):
    img = np.array(img, dtype='float32') / 255.
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]
    img = torch.from_numpy(img)
    img = Variable(img).cuda()
    return img

def resize_image(img, scale_coefficient):
    width = int(img.shape[1] * scale_coefficient)
    height = int(img.shape[0] * scale_coefficient)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def random_crop(img, mask, gt):
    row, col = 480, 720
    size = 224
    crop_img = np.zeros((size, size, 3), dtype=np.float32)
    crop_mask = np.zeros((size, size, 1), dtype=np.float32)
    crop_gt = np.zeros((size, size, 3), dtype=np.float32)
    x, y = random.randint(0, row - size), random.randint(0, col - size)
    crop_img = img[x: x + size, y: y + size]
    crop_mask = mask[x: x + size, y: y + size]
    crop_gt = gt[x: x + size, y: y + size]
    return crop_img, crop_mask, crop_gt


# weights initialize
def weights_inits(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def handle_labels(back_ground_truth):
    label = prepare_img_to_tensor(back_ground_truth)
    label_2 = prepare_img_to_tensor(resize_image(back_ground_truth, 0.5))
    label_4 = prepare_img_to_tensor(resize_image(back_ground_truth, 0.25))
    return label_4, label_2, label

def train():
    if previous_epoch != 0:
        pre_gen_model_path = f'./trains_out/{previous_epoch}_gen.pkl'
        pre_dis_model_path = f'./trains_out/{previous_epoch}_dis.pkl'
        gen.load_state_dict(pre_gen_model_path)
        dis.load_state_dict(pre_dis_model_path)
        gen.train()
        dis.train()
    else:
        # gen.apply(weights_inits)
        # dis.apply(weights_inits)
        gen.train()
        dis.train()

    loss = nn.MSELoss()
    criterion = nn.BCELoss()
    input_list = sorted(os.listdir(args.input_dir))
    gt_list = sorted(os.listdir(args.gt_dir))
    mask_list = sorted(os.listdir(args.mask_dir))
    psnr_max = 0
    logging.info(f"AttentionGAN🚀 torch{torch.__version__} device: cuda:{args.gpus}({d.name}, {d.total_memory / 1024 ** 2}MB)\n")

    for e in range(previous_epoch + 1, args.epoch):
        epoch_loss = 0
        ma_loss = 0
        ms_loss = 0
        p_loss = 0
        class_loss_real = 0
        class_loss_fake = 0
        gen_loss = 0
        dis_loss = 0

        for i in tqdm(range(len(input_list))):

            img = cv2.imread(args.input_dir + input_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])
            mask = cv2.imread(args.mask_dir + mask_list[i], cv2.IMREAD_GRAYSCALE)
            mask = mask[:, :, np.newaxis]

            img, mask, gt = random_crop(img, mask, gt)

            mask = prepare_img_to_tensor(mask)
            img = prepare_img_to_tensor(img)

            # ----------------------------
            # Discriminatior
            # ----------------------------
            trainable(gen, False)
            trainable(dis, True)

            batch_fake0 = gen(img)
            batch_fake = batch_fake0[-1]
            batch_real = prepare_img_to_tensor(gt)

            # real loss
            result_real = dis(batch_real)
            label_real = Variable(torch.ones(result_real[1].shape)).cuda()
            real_loss = criterion(result_real[1], label_real)
            class_loss_real += real_loss
            # fake loss
            result_fake = dis(batch_fake)
            label_fake = Variable(torch.zeros(result_fake[1].shape)).cuda()
            fake_loss = criterion(result_fake[1], label_fake)
            class_loss_fake += fake_loss
            # MAP loss
            label_map_real = Variable(torch.ones(result_real[0].shape)).cuda()
            MAP_loss = 0.05 * (loss(result_real[0], label_map_real) + loss(result_fake[0], batch_fake0[0][-1]))
            # Dis loss
            loss_D = real_loss + fake_loss + MAP_loss
            dis_loss += loss_D
            # Back Propagation D
            opt_d.zero_grad()
            loss_D.backward()
            opt_d.step()

            # ----------------------------
            # Generator
            # ----------------------------
            trainable(gen, True)
            trainable(dis, False)

            m_list, data_frame1, data_frame2, data_frame3 = gen(img)
            label_frame1, label_frame2, label_frame3 = handle_labels(gt)

            data_frame4 = vgg16(data_frame3)
            label_frame4 = vgg16(label_frame3)

            # Multiscale loss
            multi_scale_loss = 0.6 * loss(data_frame1, label_frame1) + 0.8 * loss(data_frame2, label_frame2) + loss(
                data_frame3, label_frame3)
            ma_loss += multi_scale_loss
            # Perceptual loss
            perceptual_loss = loss(data_frame4[0], label_frame4[0]) + 0.6 * loss(data_frame4[1],
                                                                                 label_frame4[1]) + 0.4 * loss(
                data_frame4[2], label_frame4[2]) + 0.2 * loss(data_frame4[3], label_frame4[3])
            p_loss += perceptual_loss
            # Attention loss
            mask_loss = 0
            for i in range(len(m_list)):
                mask_loss += args.theta ** (len(m_list) - i - 1) * loss(m_list[i], mask)
            ms_loss += mask_loss
            # GAN loss
            mask_d, result = dis(data_frame3)
            label_zeros = Variable(torch.zeros(result.shape)).cuda()
            gan_loss = criterion(result, label_zeros) * -0.01
            # Gen loss
            loss_G = multi_scale_loss + perceptual_loss + mask_loss + gan_loss
            gen_loss += loss_G
            # Back Propagation G
            opt_g.zero_grad()
            loss_G.backward()
            opt_g.step()

            # Total loss
            total_loss = loss_D + loss_G
            epoch_loss += total_loss
        print(f"Epoch:{e}/{args.epoch}, gen_loss:{gen_loss / len(input_list):.4f}, adv_loss:{epoch_loss / len(input_list):.4f}")
        print("Testing....")
        test_input_list = sorted(os.listdir(args.test_input_dir))
        test_gt_list = sorted(os.listdir(args.test_gt_dir))
        num = len(test_input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            # print('Processing image: %s' % (test_input_list[i]))
            img = cv2.imread(args.test_input_dir + test_input_list[i])
            gt = cv2.imread(args.test_gt_dir + test_gt_list[i])
            img = align_to_four(img)
            gt = align_to_four(gt)

            #----validation-----
            img = np.array(img, dtype='float32') / 255.
            img = img.transpose((2, 0, 1))
            img = img[np.newaxis, :, :, :]
            img = torch.from_numpy(img)
            img = Variable(img).cuda()
            out = gen(img)[-1]
            out = out.cpu().data
            out = out.numpy()
            out = out.transpose((0, 2, 3, 1))
            out = out[0, :, :, :] * 255.

            result = out
            result = np.array(result, dtype='uint8')

            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)

            # print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / num, cumulative_ssim / num))

        #Save
        if cumulative_psnr >= psnr_max:
            torch.save(gen.state_dict(), f'./trains_out/best_gen.pkl')
            torch.save(dis.state_dict(), f'./trains_out/best_dis.pkl')
            psnr_max = cumulative_psnr

if __name__ == '__main__':
    # Default Settings
    args = get_args()

    # GPU Settings & info
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    d = torch.cuda.get_device_properties(f'cuda:{args.gpus}')

    # Training Setting
    previous_epoch = args.pre_epoch

    # Training info
    if not os.path.exists(args.snapshot):
        os.mkdir(args.snapshot)
    sys.stdout = Logger(os.path.join(args.snapshot, date_modified() + '.log'))

    #---------------------------------------------------
    gen = Generator().cuda()
    dis = Discriminator().cuda()
    vgg16 = vgg(vgg_init(args.model_weights))
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.99))
    opt_d = torch.optim.Adam(dis.parameters(), lr=args.lr, betas=(0.5, 0.99))

    #---------------------------------------------------
    train()