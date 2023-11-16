# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/11/2
# User      : WuY
# File      : filter.py
# description : 对膜电位或者电流进行滤波，减少信息的损失
# 将各种用于神经网络的`滤波器`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import correlate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ASFilter(nn.Module):
    """
    适应性突触滤波器(Adaptive synaptic filter, ASF)
    reference: https://doi.org/10.1016/j.neunet.2023.06.019
    equ:
        \left. \begin{array}  {l}  {  \delta_{asf}  ( i ^ { ( t ) } ) = \frac { u_{thresh}^{( t )} } { 1 + e ^ { (\sigma_t  ) } } }
        \\ {  \quad \sigma_t = - \alpha_{asf} \frac {  i ^ { ( t ) }  } { u_{thresh}^{( t )} } + \beta_{asf}  } \end{array} \right.
    description:
        ASF 通过非线性函数调整电流，使电流更有可能集中在阈值或静息电位附近。
        将电流集中到阈值将导致对神经元的更多竞争。更多的竞争将有助于避免神经元的主导地位。
        而电流接近静息电位将减少低电流强度神经元发射时产生的噪音。
    """
    def __init__(self, alpha_asf=8., beta_asf=3.2):
        """
        args:
            alpha_asf: 系数, 控制滤波器的功能
            beta_asf: 系数, 控制滤波器的功能
        """
        super().__init__()
        self.alpha_asf = alpha_asf
        self.beta_asf = beta_asf

    def forward(self, current, thre):
        """
        args:
            current: 卷积或全连接后的电流(B,C,H,W)/(B,C)
            thre: 阈值
        return：
            current_ASF: 滤波后的电流
        """
        current = current.clamp(min=0)  # 文章中并没写明裁剪电流(需要调试,影响收敛速度)
        current_ASF = thre / (1 + torch.exp(-(current - self.beta_asf * thre / self.alpha_asf) * (self.alpha_asf / thre)))
        return current_ASF


class DoG:
    """
    高斯差分滤波器(Difference of Gaussians, DoG)
    reference: DOI: 10.1016/j.neunet.2017.12.005
    用法，可以放在ToTensor()前面，输入的必须是PIL图像
    """
    def __init__(self, size, std1=1, std2=2):
        """
        args:
            size: 滤波器的尺度(size x size)
            std1: 滤波器一的标准差
            std2: 滤波器二的标准差
        """
        self.DoG_filter(size, std1=std1, std2=std2)

    def __call__(self, x):
        """
        输入张量数据，使用卷积滤波
        x: 输入图片数据
        """
        img = np.asarray(x.getdata(), dtype=np.float64).reshape((x.size[1], x.size[0]))
        self.img_in = img / np.amax(img)
        # Apply filter
        img = correlate(img, self.filt, mode='constant')
        # Border
        border = np.zeros(img.shape)
        border[5:-5, 5:-5] = 1.
        img = img * border
        # Threshold
        img = (img >= 15).astype(int) * img
        img = np.abs(img)  # Convert -0. to 0.

        self.img_out = img / np.amax(img)
        return self.img_out

    def DoG_filter(self, size, std1=1, std2=2):
        """
        Generates a filter window of size size x size with std of s1 and s2
        args:
            size: 滤波器的尺度(size x size)
            std1: 滤波器一的标准差
            std2: 滤波器二的标准差
        """
        r = np.arange(size) + 1
        self.x, self.y = np.meshgrid(r, r)
        d2 = (self.x - size / 2. - 0.5) ** 2 + (self.y - size / 2. - 0.5) ** 2
        # self.filt = 1/np.sqrt(2*np.pi) * (1/std1 * np.exp(-d2/(2*(std1**2))) - 1/std2 * np.exp(-d2/(2*(std2**2))))
        # 二维高斯函数(两个录波器相减)
        self.filt = 1 / (2 * np.pi) * (1 / (std1 ** 2) * np.exp(-d2 / (2 * (std1 ** 2)))
                                       - 1 / (std2 ** 2) * np.exp(-d2 / (2 * (std2 ** 2))))
        self.filt -= np.mean(self.filt)
        self.filt /= np.amax(self.filt)

    def readim(self, path_img, img_size=(250, 160)):
        """
        读取图片并转化为numpy
        """
        # 打开原始图像
        with Image.open(path_img) as img:
            img = img.convert('L')  # 转换为灰度图
            img = img.resize(img_size)  # 调整图像大小为新的尺寸
            self(img)

    def plotFilter(self):
        """
        输出滤波器图像
        """
        plt.figure(figsize=(12, 6))
        ax = plt.subplot(1, 1, 1, projection='3d')
        ax.plot_surface(self.x, self.y, self.filt, cmap="viridis")
        plt.tight_layout()
        plt.show()

    def plotImageDoG(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(self.img_in, cmap='gray')  # , cmap="viridis"
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(self.img_out, cmap='gray_r')
        plt.colorbar()
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    # ASF = ASFilter()
    # current = torch.tensor(0.1)
    # print(ASF(current, torch.tensor(0.5)))
    DoG = DoG(10)
    # DoG.plotFilter()
    DoG.readim(r"C:\Users\67642\Desktop\dog.jpg")
    DoG.plotImageDoG()
    # print(DoG.filt)
