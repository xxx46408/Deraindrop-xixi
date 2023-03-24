## Pytorch implementation: Attentive Generative Adversarial Network for Raindrop Removal from A Single Image (CVPR'2018)

Reference: [[rui1996]](https://github.com/rui1996/DeRaindrop) 

PaperLink: [[Link]](https://arxiv.org/abs/1711.10098)

## Description:

This project is unofficial pytorch implementation version of training parts. 

## Prerequisites:
1. Linux
2. Python 3.8
3. NVIDIA GPU + CUDA CuDNN (CUDA 11.6)
4. Pytorch 1.12.0


## Installation:
1. Clone this repo
2. Install PyTorch and dependencies from http://pytorch.org

（**Note:** the code is suitable for PyTorch 1.12.0）
## Usage 
1. Download `vgg16-397923af.pth` beforehand 
2. Generating Binary mask from clean images and degraded images by using `./uilt/mask_gen.py`
3. Train your own weight by using `train.py`
3. The best result of my implementation is under `./weights/best_gen.pkl` ,and the test results are below:

   **test_a :** PSNR-29.6134 SSIM-0.9025
   
   **test_b :** PSNR-24.5358 SSIM-0.7953

## Dataset
The whole dataset can be find in author pages(https://github.com/rui1996/DeRaindrop)

## Results
![Image text](results/demoresult.png)
