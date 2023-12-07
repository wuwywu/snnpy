# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2023/12/6
# User      : WuY
# File      : createConn.py
# 将各种用于网络的`连接矩阵`集合到这里



def alltoall(in_num=10, out_num=None, selfconn=False):
    """
    全连接all-to-all
    创建一个全 1 矩阵
    args:
        in_num:     输入节点数
        out_num:    输出节点数
        selfconn:   是否自连接(对角线上为1/0)
    return:
        matrix:     连接矩阵shape(out_num, in_num)
    """
    if out_num is None:
        out_num = in_num
    matrix = np.ones((out_num, in_num), dtype=int)
    if selfconn:
        # 将对角线上的元素设置为 0
        np.fill_diagonal(matrix, 0)

    return matrix





