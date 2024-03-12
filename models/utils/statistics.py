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


# Kuramoto Order parameter
class cal_kop:
    """
    通过记录放电开始的时间以及放电神经元的序号
    计算 Kuramoto Order parameter
    dt: 计算的时间步长
    """
    def __init__(self, dt=0.01):
        self.dt = dt
        self.pltPlace = []      # neuron label
        self.pltTime = []       # time spike

    def __call__(self, flaglaunch, t):
        """
        flaglaunch: 放电开始标志
        t: 运行时间
        """
        flag = flaglaunch.astype(int)
        firingPlace = list(np.where(flag > 0)[0])  # 放电的位置
        lens = len(firingPlace)  # 放电位置的数量
        self.pltPlace.extend(firingPlace)  # 记录放电位置
        self.pltTime.extend([t] * lens)  # 记录放电时间

    def return_kop(self):
        spkt = np.array(self.pltTime.copy())
        spkid = np.array(self.pltPlace.copy())
        dt = self.dt
        # 1、设置一个阈值，一个神经元必须要超过这个分的数量才计算
        thrs_spks = 10
        unique_neurons, counts = np.unique(spkid, return_counts=True)
        filt_neurons = unique_neurons[counts > thrs_spks]  # 过滤后的神经元

        if filt_neurons.shape[0] != unique_neurons.shape[0]:
            print("存在神经元峰的计数不足{}".format(thrs_spks))

        # 2、找到第一个峰和最后一个峰的时间，并计算最大的首峰和最小的尾峰时间段
        first_spikes = []
        last_spikes = []
        for idx in filt_neurons:
            first_spikes.append(spkt[np.where(spkid == idx)[0][0]])
            last_spikes.append(spkt[np.where(spkid == idx)[0][-1]])

        first_last_spk = np.max(first_spikes)  # define the start of interval
        last_first_spk = np.min(last_spikes)  # define the end of interval

        # 3、计算每个神经元的相位
        ttotal = spkt[-1] - spkt[0]
        time_vec = np.linspace(spkt[0], spkt[-1], int(ttotal / dt))

        phase = np.ones((len(filt_neurons), len(time_vec))) * -1
        peak_id = np.ones((len(filt_neurons), len(time_vec))) * -1  # 记录相位属于的峰值编号

        for z, neuron_label in enumerate(filt_neurons):
            idx_individual_spikes = np.where(spkid == neuron_label)[0]  # 神经元放电的位置
            individual_spkt = spkt[idx_individual_spikes]   # 神经元放电的时间
            for i, t in enumerate(individual_spkt[:-1]):
                ti = np.where(time_vec >= t)[0][0]  # t_n
                tf = np.where(time_vec >= individual_spkt[i + 1])[0][0] # t_n+1
                phase[z][ti:tf] = np.linspace(0, 2. * np.pi, (tf - ti))
                peak_id[z][ti:tf] = i + 1  # 记录相位属于的峰值编号

        # 剪切出定义的相位的区间
        idxs = np.where((time_vec > first_last_spk) & (time_vec < last_first_spk))[0]
        phase = phase[:, idxs]
        peak_id = peak_id[:, idxs]  # 剪切出定义的区间
        peak_id -= peak_id[:, :1]

        # 计算 Kuramoto Order parameter
        kuramoto = np.abs(np.mean(np.exp(1j * phase), axis=0))

        # 若需要则以输出以下量
        # 1、随时间变化的 Kuramoto Order parameter 存在 kuramoto
        # 2、每个神经元的相位变化存在 phase
        # 3、计算的时间为 first_last_spk -- last_first_spk
        return np.mean(kuramoto), kuramoto, phase, peak_id, (first_last_spk, last_first_spk)


# 计算尖峰序列的信息熵 entropy 和两个序列的互信息 mutual information
class cal_information:
    """
    用于计算尖峰序列的信息熵 entropy 和两个序列的互信息 mutual information
    time_s: 计算开始的时间
    time_e: 计算结束的时间
    bin_size: bin，一个时间窗口，在这个窗口中计算峰的个数 (ms), 非常重要的参数需要调节选取
    """
    def __init__(self, time_s, time_e, bin_size=15):
        self.time_s = time_s
        self.time_e = time_e
        self.num_bins = int(np.ceil((time_e-time_s) / bin_size))    # 将总时间段分为格子数
        self.pltPlace = []  # neuron label
        self.pltTime = []  # time spike

    def __call__(self, flaglaunch, t):
        """
        flaglaunch: 放电开始标志(用于记录峰)
        t: 运行时间
        """
        flag = flaglaunch.astype(int)
        firingPlace = list(np.where(flag > 0)[0])  # 放电的位置
        lens = len(firingPlace)  # 放电位置的数量
        self.pltPlace.extend(firingPlace)  # 记录放电位置
        self.pltTime.extend([t] * lens)  # 记录放电时间

    def return_info(self):
        spkt = np.array(self.pltTime.copy())
        spkid = np.array(self.pltPlace.copy())
        unique_neurons = np.unique(spkid)
        spike_count_list = []   # 保存每个神经元的尖峰计数
        entropy = []            # 保存每个神经元的信息熵
        for idx in unique_neurons:
            spike_times = spkt[np.where(spkid == idx)[0]]
            spike_count, _ = np.histogram(spike_times, bins=self.num_bins, range=(self.time_s, self.time_e))
            spike_count_list.append(spike_count)

            entropy.append(self._entropy(spike_count))

        # 计算所有的互信息
        MI = np.zeros((len(unique_neurons), len(unique_neurons)))
        for i, idx in enumerate(unique_neurons):
            for j, idy in enumerate(unique_neurons):
                MI[i, j] = self._mutual_information(spike_count_list[i], spike_count_list[j])

        return entropy, MI

    def _entropy(self, spike_counts):
        """
        计算信息熵
        spike_counts: 一个尖峰序列计数的数组
        return:
            entropy: 尖峰序列的信息熵信息熵
        """
        # 计算概率分布
        values, counts = np.unique(spike_counts, return_counts=True)
        probabilities = counts / sum(counts)
        # 计算信息熵
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))

        return entropy

    def _mutual_information(self, spike_counts1, spike_counts2):
        """
        计算两组尖峰序列的互信息
        spike_counts1: 第一组序列的尖峰计数
        spike_counts2: 第二组序列的尖峰计数
        return:
            MI: 互信息 (mutual information)
        """
        # 计算联合概率分布
        joint_hist, _, _ = np.histogram2d(spike_counts1, spike_counts2, bins=self.num_bins)
        p_xy = joint_hist / np.sum(joint_hist)  # 联合概率分布
        p_x = np.sum(p_xy, axis=1, keepdims=True)  # 边缘概率分布，保持维度用于后续广播操作
        p_y = np.sum(p_xy, axis=0, keepdims=True)  # 边缘概率分布，保持维度用于后续广播操作

        # 使用矩阵运算计算互信息
        with np.errstate(divide='ignore', invalid='ignore'):
            MI_matrix = p_xy * np.log2(p_xy / (p_x @ p_y) + 1e-12)  # @符号用于矩阵乘法
            MI_matrix[np.isnan(MI_matrix)] = 0  # 处理NaN值
            MI = np.sum(MI_matrix)

        return MI

