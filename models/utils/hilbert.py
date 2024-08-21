# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/8/20
# User      : WuY
# File      : hilbert.py
# 使用希尔伯特变换求出 幅度， 频率， 瞬时频率。

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


def tohilbert(signal, time=None):
    """
    signal : 输入信号[ndim, Ntime]
    time   : 时间 [N, ] (如果不输入则无法求出 瞬时频率)
    """
    # 进行希尔伯特变换
    analytic_signal = hilbert(signal)

    # 计算幅度
    amplitude = np.abs(analytic_signal)

    # 计算相位
    phase = np.angle(analytic_signal)
    phase = (phase + np.pi) / (2*np.pi)

    # 计算瞬时频率
    if time is not None:
        instantaneous_frequency = np.diff(np.unwrap(phase)) / (2 * np.pi * np.diff(time))
    else : instantaneous_frequency = None

    return amplitude, phase, instantaneous_frequency


def calculate_complete_phases(phases):
    """
    计算完整的相位变化：
        2*pi*N + phase(0-1)
    注意：
        希尔伯特变换求出相位后，将首尾的误差去掉，截取中间的有效部分
    """
    complete_phases = []
    complete_phases.append(phases[0] * 2 * np.pi)
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i - 1]
        if diff < -0.9:
            diff += 1
        complete_phase = complete_phases[i - 1] + diff * 2 * np.pi
        complete_phases.append(complete_phase)
    return np.array(complete_phases)


if __name__ == "__main__":
    # 生成多个示例信号
    t = np.linspace(0, 10, 1000)
    # signal1 = np.sin(2 * np.pi * 2 * t)
    # signal2 = np.cos(2 * np.pi * 3 * t)
    signal1 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    signal2 = np.cos(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 4 * t)

    signals = np.array([signal1, signal2])

    amplitudes, phases, instantaneous_frequencies = tohilbert(signals, t)

    # 绘制结果
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))

    axs[0, 0].plot(t, signals[0])
    axs[0, 0].set_title('Signal 1')
    axs[0, 1].plot(t, signals[1])
    axs[0, 1].set_title('Signal 2')

    axs[1, 0].plot(t, amplitudes[0])
    axs[1, 0].set_title('Amplitude of Signal 1')
    axs[1, 1].plot(t, amplitudes[1])
    axs[1, 1].set_title('Amplitude of Signal 2')

    axs[2, 0].plot(t, phases[0])
    axs[2, 0].set_title('Phase of Signal 1')
    axs[2, 1].plot(t, phases[1])
    axs[2, 1].set_title('Phase of Signal 2')

    axs[3, 0].plot(t[:-1], instantaneous_frequencies[0])
    axs[3, 0].set_title('Instantaneous Frequency of Signal 1')
    axs[3, 1].plot(t[:-1], instantaneous_frequencies[1])
    axs[3, 1].set_title('Instantaneous Frequency of Signal 2')

    plt.tight_layout()
    plt.show()
