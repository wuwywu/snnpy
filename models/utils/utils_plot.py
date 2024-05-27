device = "cpu"
import numpy
if device == "gpu":
    import cupy
    np1 = numpy
    np = cupy
else:   
    np1 = numpy
    np = numpy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps  

custom_colors = {
                'blue': (0, 0.45, 0.7),
                'red': (0.57, 0, 0),
                'orange': (0.9, 0.6, 0),
                'vermilion': (0.8, 0.4, 0),
                'bluish_green': (0, 158./255, 115./255),
                'yellow': (0.95, 0.9, 0.25),
                'sky_blue': (86./255, 180./255, 233./255),
                'pink': (0.8, 0.6, 0.7),
                'light_blue': (109./255, 182./255, 1.),
                'special_green': (88./255, 180./255, 109./255),
                'teacher_green': (0., 146./255, 146./255)
                }


# 图像色彩梯度处理
class Cmap:
    '''
    colormap的处理
    颜色的获取可去查看：
        1、https://matplotlib.org/2.0.2/gallery.html#color
        2、https://mycolor.space/(获取颜色的三原色版本)
    '''
    @staticmethod
    def hex2RGB(*args):
        '''
        例如：'#d16ba5' --> (0.8196078431372549, 0.4196078431372549, 0.6470588235294118)
        '''
        custom_colors = []
        for rgb_color in args:
            custom_colors.append(mcolors.hex2color(rgb_color))
        return custom_colors
    
    @staticmethod
    def custom_cmap(custom_colors):
        '''
        从一个RGB颜色列表中获取cmap
        如：自定义jet颜色
            cmap_colors = [
                (0, 0, 0.5),   # 深蓝色
                (0, 0, 1),     # 蓝色
                (0, 1, 1),     # 青色
                (1, 1, 0),     # 黄色
                (1, 0, 0),     # 红色
                (0.5, 0, 0)    # 深红色
            ]
        '''
        return mcolors.LinearSegmentedColormap.from_list('custom_cmap', custom_colors)
    
    @staticmethod
    def get_cmap(cmap='jet'):
        '''
        # Have colormaps separated into categories:
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
        '''
        return colormaps.get_cmap(cmap)
    

# 画放电点图
class plot_firing_raster:
    def __init__(self, startTime=None, endTime=None):
        self.startTime = startTime
        self.endTime = endTime
        self.pltPlace = []
        self.pltTime = []

    def __call__(self, flag, t):
        # 输入放电标志和运行时间
        if self.startTime is not None:
            if t>=self.startTime and t<=self.endTime:
                flag = flag.astype(int)
                firingPlace = list(np.where(flag>0)[0]) # 放电的位置
                lens = len(firingPlace)                 # 放电位置的数量
                self.pltPlace.extend(firingPlace)       # 记录放电位置
                self.pltTime.extend([t]*lens)           # 记录放电时间
        else:
            flag = flag.astype(int)
            firingPlace = list(np.where(flag>0)[0]) # 放电的位置
            lens = len(firingPlace)                 # 放电位置的数量
            self.pltPlace.extend(firingPlace)       # 记录放电位置
            self.pltTime.extend([t]*lens)           # 记录放电时间

    def plot_raster(self, markersize=.05, color=custom_colors['vermilion']):
        # plt.scatter(self.pltTime, self.pltPlace, 
        #             s=.8, c=custom_colors['light_blue'])
        plt.plot(self.pltTime, self.pltPlace, "o", 
                 markersize=markersize, color=color)
    
    def reset(self):
        self.__init__(self.startTime, self.endTime)
        

# 画线条图
class plot_line:
    def __init__(self):
        self.x = []
        self.y = []
    
    def __call__(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def plot(self):
        plt.plot(self.x, self.y)


# 画误差阴影图
def plot_errorFill(ax, x, y_mean, y_error,
                  linewidth=2.5, color="b", alpha=0.2, label=None):
    """
    ax      : 画图的轴体
    x       : x轴的数据
    y_mean  : y轴平均值数据
    y_error : y轴误差数据
    linewidth  : 线的宽度
    color   : 颜色（默认蓝色）
    alpha   : 阴影的透明度
    """
    ax.plot(x, y_mean, "-", linewidth=linewidth, color=color, label=label)
    ax.fill_between(x, y_mean-y_error, y_mean+y_error, color=color, alpha=alpha)


# 画误差棒图
def plot_errorbar(ax, x, y_mean, y_error, color='b', capsize=5, 
                  linewidth=2.5, markersize=5, label=None):
    """
    ax      : 画图的轴体
    x       : x轴的数据
    y_mean  : y轴平均值数据
    y_error : y轴误差数据
    label   : 标签
    color   : 颜色（默认蓝色）
    capsize : 误差棒上下两端条的大小
    linewidth  : 线的宽度
    markersize : 标记的宽度（默认为"o"的大小）
    """
    ax.errorbar(x, y_mean, yerr=y_error, capsize=capsize, 
                linestyle='-', linewidth=linewidth, 
                marker='o', markersize=markersize, 
                label=label, color=color, alpha=0.8)


# 设置子图属性
def set_ax(ax, title=None, title_x=None, 
           xlim=None, xlabel=None, xlabel_coords=None, 
           ylim=None, ylabel=None, ylabel_coords=None, 
           font_size=None):
    '''
    --> 若需要单独设置，将命令单独拿出去运行(查找命令改变更多)
    ax : 轴体
    title : 标题
    title_x : 标题的水平位置
    xlim : x轴的范围
    xlabel : x轴标签
    xlabel_coords : x轴标签的坐标(x, y)
    ylim : y轴的范围
    ylabel_coords : y轴标签的坐标(x, y)
    ylabel : y轴标签
    font_size = [tick_size, label_size, title_size]
    '''
    # 设置字体
    font = {'family': 'Times New Roman', 'style': 'italic'}
    font1 = {'family': 'Times New Roman'}

    # 设置字体大小
    tick_size = 12           # 刻度大小
    label_size = 22         # 标签大小
    title_size = 24         # 标题大小
    if font_size is not None: font_size = font_size
    else: font_size = [tick_size, label_size, title_size]

    if title is not None:   ax.set_title(title, fontsize=font_size[2], fontdict=font1) # 调整标题垂直位置使用：pad=20
    if title_x is not None:   ax.title.set_position([title_x, 1])  # 调整标题位置

    if xlim is not None:    ax.set_xlim(*xlim)
    if xlabel is not None:    ax.set_xlabel(xlabel, fontsize=font_size[1], fontdict=font)
    if xlabel_coords is not None:    ax.xaxis.set_label_coords(*xlabel_coords) # 设置x轴与x标签的相对位置
    
    if ylim is not None:    ax.set_ylim(*ylim)
    if ylabel is not None:    ax.set_ylabel(ylabel, fontsize=font_size[1], fontdict=font)
    if ylabel_coords is not None:    ax.yaxis.set_label_coords(*ylabel_coords) # 设置y轴与y标签的相对位置

    ax.tick_params(labelsize=font_size[0])       # axis={'x', 'y', 'both'},
                         

# 调节轴体的相对位置（上下左右移动）
def move_ax(ax, m_lr=0, m_ud=0):
    '''
    ax :  轴体
    m_lr: 向左向右移动(+:右, -:左) 0-1的相对位置
    m_ud: 向上向下移动(+:上, -:下) 0-1的相对位置

    注意：
        1、画图中使用了fig.tight_layout, fig.subplots_adjust, 在后面使用移动
        2、调节颜色棒的位置，输入colorbar.ax的轴体

    '''
    position = ax.get_position()    # 获取子轴相对位置

    # 移动子图位置（向右和向上移动）
    mrl = m_lr
    mud = m_ud
    new_x0 = position.x0 + mrl  # 向右移动
    new_y0 = position.y0 + mud  # 向上移动
    new_x1 = position.x1 + mrl  # 向右移动
    new_y1 = position.y1 + mud  # 向上移动

    ax.set_position([new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0])


if __name__=="__main__":
    fig, ax = plt.subplots()
    set_ax(ax, title=None, title_x=None, 
           xlim=None, xlabel=None, xlabel_coords=None, 
           ylim=None, ylabel=None, ylabel_coords=None, 
           font_size=None)
    plt.show()


