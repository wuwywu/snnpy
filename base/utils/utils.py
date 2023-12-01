# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/21
# User      : WuY
# File      : utils.py
# 将各种用于神经网络的`通用工具`集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中\
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def setup_seed(seed=3407):
    """
    为CPU，GPU，所有GPU，numpy，python设置随机数种子，并禁止hash随机化
    :param seed: 随机种子值
    :return:
    """
    torch.manual_seed(seed)             # 给cpu设置随机种子
    torch.cuda.manual_seed(seed)        # 给当前gpu设置随机种子
    torch.cuda.manual_seed_all(seed)    # 给所有gpu设置随机种子
    np.random.seed(seed)                # 给numpy设置随机种子
    random.seed(seed)                   # 给自带随机函数设置随机种子

    # 该标记可启用 cudnn 自动调整器，它能找到使用的最佳算法
    # 针对特定配置。(该模式适用于输入尺寸没有变化的情况）
    torch.backends.cudnn.benchmark = False  # 禁止自动优化配置带来的随机
    # 该标志只允许使用与基准不同的确定性 cudnn 算法。
    torch.backends.cudnn.deterministic = True   # 获得确定性（牺牲最佳性能）

    os.environ['PYTHONHASHSEED'] = str(seed)    # 禁止hash随机化

