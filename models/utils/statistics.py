# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/2/28
# User      : WuY
# File      : statistics.py
# 用于研究的统计量

# device = "gpu"
device = "cpu"
# from settings import *
import numpy
if device == "gpu":
    import cupy
    np1 = numpy
    np = cupy
else:
    np1 = numpy
    np = numpy
import matplotlib.pyplot as plt


# 计算同步因子
class cal_synFactor:
    """
    计算变量的同步因子
    Tn: 计算次数(int)，Time/dt
    num: 需要计算变量的数量

    描述：一个用于量化同步的归一化统计量。越趋近与1，同步越好；越趋近与0，同步越差（范围在0-1之间）
    """
    def __init__(self, Tn, num):
        self.Tn = Tn  # 计算次数
        self.n = num  # 矩阵大小
        self.count = 0  # 统计计算次数
        # 初始化计算过程
        self.up1 = 0
        self.up2 = 0
        self.down1 = np.zeros(num)
        self.down2 = np.zeros(num)

    def __call__(self, x):
        F = np.mean(x)
        self.up1 += F * F / self.Tn
        self.up2 += F / self.Tn
        self.down1 += x * x / self.Tn
        self.down2 += x / self.Tn
        self.count += 1  # 计算次数叠加

    def return_syn(self):
        if self.count != self.Tn:
            print(f"输入计算次数{self.Tn},实际计算次数{self.count}")
        down = np.mean(self.down1 - self.down2 ** 2)
        if down > -0.000001 and down < 0.000001:
            return 1.
        up = self.up1 - self.up2 ** 2

        return up / down

    def reset(self):
        self.__init__(self.Tn, self.n)


class cal_cv:
    """
    N : 建立神经元的数量
    th_up: 判断发放开始阈值
    th_down=0: 判断发放结束阈值
    max: 初始最大值

    变异系数 The coefficient of variation (CV)
        CV=1，会得到泊松尖峰序列（稀疏且不连贯的尖峰）。
        CV<1，尖峰序列变得更加规则，并且对于周期性确定性尖峰，CV 趋近与0。
        CV>1，对应于比泊松过程变化更大的尖峰点过程。
    """
    def __init__(self, N, th_up=0, th_down=0, max=-70):
        '''
        HH可以设置为：th_up=0, th_down=0, max=-70.0
        '''
        self.num = N  # 节点数量
        self.reset_init(th_up=th_up, th_down=th_down, max=max)

    def reset_init(self, th_up, th_down, max):
        self.th_up = th_up      # 阈上值
        self.th_down = th_down  # 阈下值
        self.max_init = max  # 初始最大值
        self.max = max+np.zeros(self.num)   # 初始变化最大值
        self.nn = np.zeros(self.num)        # 记录每个节点的ISI的个数
        self.flag = np.zeros(self.num)      # 放电标志
        self.T_pre = np.zeros(self.num)     # 前峰时间
        self.T_post = np.zeros(self.num)    # 后峰时间
        self.sum = np.zeros(self.num)       # 每个节点ISI的和
        self.sum2 = np.zeros(self.num)      # 每个节点ISI平方的和

    def __call__(self, t, mem):
        """
        t: 运行时间
        mem: 输入需要计算的变量（eg. 膜电位）
        在非人工神经元中，计算神经元的spiking
        """
        # -------------------- 放电开始 --------------------
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))  # 放电开始的位置
        self.flag[firing_StartPlace] = 1  # 放电标志改为放电
        # -------------------- 放电期间 --------------------
        firing_Place = np.where((mem > self.max) & (self.flag == 1))  # 放电期间并且还没有到达峰值
        self.max[firing_Place] = mem[firing_Place]
        self.T_post[firing_Place] = t
        #  -------------------- 放电结束 -------------------
        firing_endPlace = np.where((mem < self.th_down) & (self.flag == 1))  # 放电结束的位置
        firing_endPlace2 = np.where((mem < self.th_down) & (self.flag == 1) & (self.nn>2))  # 放电结束的位置2
        self.flag[firing_endPlace] = 0  # 放电标志改为放电
        self.nn[firing_endPlace] += 1   # 结束放电ISI数量+1

        ISI = self.T_post[firing_endPlace2]-self.T_pre[firing_endPlace2]
        self.sum[firing_endPlace2] += ISI
        self.sum2[firing_endPlace2] += ISI*ISI

        self.T_pre[firing_endPlace] = self.T_post[firing_endPlace]
        self.max[firing_endPlace] = self.max_init

    def return_cv(self):
        """
        return:
            cv: 每个节点的CV
            cv_mean: 平均CV
        """
        aver = self.sum/(self.nn-2)
        var = (self.sum2/(self.nn-2) - aver*aver)**(1/2)
        cv = var/aver
        cv_mean = cv.mean()

        return cv, cv_mean
