# wuyong@ccnu
# 一些工具

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

# noise产生器
class noise_types:
    """
    产生噪声
    size: 要产生噪声的尺度，可以是整数或元组
    delta_t: 计算步长，与算法一致
    type: 噪声类型，白噪声，色噪声 ["white", "color"]
    """
    def __init__(self, size, delta_t, type="white"):
        self.size = size
        self.dt = delta_t
        self.type = type  
        if type == "color" : self.color_init = False

    def __call__(self, D_noise=0., lam_color=0.1):
        """
        D_noise: 噪声强度
        lam_color: 相关率
        """
        if self.type=="white":   noise = np.random.normal(loc=0., scale=np.sqrt(2*D_noise*self.dt), size=self.size)  
        if self.type=="color": 
            if self.color_init is False : 
                self.noise = self.dt*np.random.normal(loc=0., scale=np.sqrt(D_noise*lam_color), size=self.size)
                self.color_init = True 
            else:
                self.noise = self.noise - self.dt*lam_color*self.noise \
                    +lam_color*np.random.normal(loc=0., scale=np.sqrt(2*D_noise*self.dt), size=self.size)
                self.noise = self.dt*self.noise

            noise = self.noise

        return noise


# 延迟存储器
class delayer:
    """
    存储参数的延迟值
    num: 需要存储的延迟量
    Tn: 延迟时长 delay/dt
    """
    def __init__(self, num, Tn):
        self.n = num                          # 延迟变量数量
        self.delayLong = Tn                   # 延迟时长    
        self.delay = np.zeros((num, Tn+1))    # 延迟存储矩阵
        self.k = 0                            # 指针
        self.delay_o = np.zeros(num)          # 需要的延迟

    def __call__(self, x):
        # 更新延迟器
        if self.k == self.delayLong+1:     
            self.k = 0
        if self.k == 0:                     
            self.delay[:, self.delayLong] = x
        else:                                       
            self.delay[:, self.k-1] = x 
        # 输出延迟值
        self.delay_o = self.delay[:, self.k]    
        self.k += 1             # 前进指针

        return self.delay_o
         
    def reset(self, Tn=None):
        if Tn is not None:
            self.__init__(self.n, Tn)


# ISI(interspike interval)
class ISIer:
    def __init__(self, th_up=0, th_down=0, max=-70):
        ''' 
        HH可以设置为
        th_up=0, th_down=0, max=-70.0
        ''' 
        self.reset_init(th_up=th_up, th_down=th_down, max=max)
        self.pltx = []
        self.plty = [] 

    def reset_init(self, th_up, th_down, max):
        self.th_up = th_up          # 阈上值
        self.th_down = th_down      # 阈下值
        self.max_init = max         # 初始最大值
        self.max = max              # 初始变化最大值
        self.flag = 0               # 放电标志
        self.nn = 0                 # 峰的个数
        self.T_pre = 0              # 前峰时间
        self.T_post = 0             # 后峰时间

    def __call__(self, x, t, y):
        # 进入之前请初始化数据
        # x:测量ISI的值，y:变化的值，t:理论运行时间
        if x>self.th_up and self.flag==0:
            self.flag = 1
        if self.flag==1 and x>self.max:
            self.max = x
            self.T_post = t
        if x<self.th_down and self.flag==1:
            self.flag = 0
            self.nn += 1
            if self.nn>2:
                ISI = self.T_post-self.T_pre
                self.pltx.append(y)
                self.plty.append(ISI)
            self.T_pre = self.T_post
            self.max = self.max_init        # 初始化

    def plt_ISI(self, markersize=.8, color="k"):
        plt.plot(self.pltx, self.plty, "o", 
                 markersize=markersize, color=color)
        
    def reset(self):
        self.reset_init(th_up=self.th_up, th_down=self.th_down 
                        ,max=self.max_init)


class spikevent:
    """
    神经元模型的峰值收集器
    N: 收集 N 个神经的尖峰事件
    """
    def __init__(self, N):
        self.N = N
        self.Tspike_list = [[] for i in range(N)]

    def __call__(self, t, spikes):
        """
        t: 模拟实时时间
        spikes: 是否尖峰的标志，如 flaglaunch(放电开启标志)
        """
        for i in range(self.N):
            if spikes[i]>0.9:
                self.Tspike_list[i].append(t)

    def pltspikes(self):
        plt.eventplot(self.Tspike_list)
