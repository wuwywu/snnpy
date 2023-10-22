# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/10/19
# User      : WuY
# File      : visualization.py
# 使用于机器学习中的`可视化`

import numpy as np
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import pandas as pd

# Random state.
RS = 20150101

def plot_tsne2d(x, colors, ax=None, num_classes=None):
    """
    t-Distributed Stochastic Neighbor Embedding（t-SNE）
    t-SNE method : 可以展示聚类能力
    referenc : Maaten, L. V. d., and Hinton, G. (2008). Visualizing data using t-SNE. J. Mach. Learn. Res. 9, 2579-2605.
    库 : sklearn.manifold.TSNE

    :param ax: 创建的轴
    :param x: 输入的feature map / spike (将数据变为[B, N]形状)
    :param colors: predicted labels 作为不同类别的颜色
    """
    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.to('cpu').numpy()
    if num_classes is None:
        num_classes=colors.max()+1  # 获取类别尺寸
    # n_components=2 指定我们希望降维到的目标维度，random_state=RS 用于确保结果的可重复性
    x = TSNE(random_state=RS, n_components=2).fit_transform(x)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    palette = np.array(sns.color_palette("hls", num_classes)) # 获取不同类别的颜色（三原色）
    if ax is None:
        fig, ax1= plt.subplots(1,1,figsize=(8, 8))
    else: 
        ax1 = ax
    sc = ax1.scatter(x[:, 0], x[:, 1], lw=0, s=25,
                    c=palette[colors.astype(np.int16)])
    legend_list = np.arange(num_classes)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_list[i], 
                             markersize=10, markerfacecolor=palette[i]) for i in range(len(legend_list))]
    legend = ax1.legend(handles=legend_handles, title=None, framealpha=0.6, fontsize=12, ) # bbox_to_anchor=(0.8, 0.8)
    legend.get_frame().set_facecolor('w')  # 设置图例背景颜色
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    ax1.axis('tight')
    # plt.grid('off')

    # plt.savefig(output_dir, facecolor=fig.get_facecolor(), bbox_inches='tight')
    # plt.show()


def plot_tsne3d(x, colors, ax=None, num_classes=None):
    """
    t-Distributed Stochastic Neighbor Embedding（t-SNE）
    t-SNE method : 可以展示聚类能力
    referenc : Maaten, L. V. d., and Hinton, G. (2008). Visualizing data using t-SNE. J. Mach. Learn. Res. 9, 2579-2605.
    库 : sklearn.manifold.TSNE

    :param ax: 创建的轴
    :param x: 输入的feature map / spike (将数据变为[B, N]形状)
    :param colors: predicted labels 作为不同类别的颜色
    """
    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.to('cpu').numpy()

    if num_classes is None:
        num_classes=colors.max()+1
    # 算法中的一个关键超参数。它影响了算法识别局部和全局结构的方式，以及生成的两维或三维嵌入的质量。
    x = TSNE(random_state=RS, n_components=3, perplexity=30).fit_transform(x)
    # sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    palette = np.array(sns.color_palette("hls", num_classes))
    if ax is None:
        fig, ax1= plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': '3d'})
    else: 
        ax1 = ax
    sc = ax1.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=20, alpha=0.8,
                    c=palette[colors.astype(np.int16)])
    legend_list = np.arange(num_classes)
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_list[i], 
                             markersize=10, markerfacecolor=palette[i]) for i in range(len(legend_list))]
    legend = ax1.legend(handles=legend_handles, title=None, framealpha=0.6, fontsize=12, ) # bbox_to_anchor=(0.5, 0.6)
    legend.get_frame().set_facecolor('w')  # 设置图例背景颜色
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(20, -120)
    ax1.axis('tight')
    # plt.savefig(output_dir, facecolor=fig.get_facecolor(), bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    # Test for T-SNE
    # x = torch.randn((100, 100))
    # y = torch.randint(low=0, high=10, size=[100])
    # fig, ax= plt.subplots(1,1,figsize=(8, 8))
    # plot_tsne2d(x, y, ax)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': '3d'})
    # plot_tsne3d(x, y, ax)
    file1 = r"./datas/save3.csv"
    df1 = pd.read_csv(file1)
    data1 = df1.values
    file2 = r"./datas/save4_labels.csv"
    df_labels = pd.read_csv(file2)
    label = df_labels.values.flatten()
    plot_tsne2d(data1, label)

    plt.show()


