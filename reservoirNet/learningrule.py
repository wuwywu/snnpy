# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/1
# User      : WuY
# File      : learningrule.py
# 用于reservoir的算法集合到这里

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# online
from base.learningrule.rls import RLS       # FORCE(Recursive least squares algorithm)
from base.learningrule.ridge import Ridge   # # 岭回归算法--> ridge regression

