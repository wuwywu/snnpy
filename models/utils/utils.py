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
    def __init__(self, size, delta_t, type="white"):
        self.size = size
        self.dt = delta_t
        self.type = type  
        if type == "color" : self.color_init = False

    def __call__(self, D_noise=0., lam_color=0.1):
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
    

# 计算同步因子
class cal_synFactor:
    def __init__(self, Tn, num):
        self.Tn = Tn    # 计算次数
        self.n = num    # 矩阵大小
        self.count = 0  # 统计计算次数
        # 初始化计算过程
        self.up1 = 0
        self.up2 = 0
        self.down1 = torch.zeros(num)
        self.down2 = torch.zeros(num)

    def __call__(self, x):
        F = torch.mean(x)
        self.up1 += F*F/self.Tn
        self.up2 += F/self.Tn
        self.down1 += x*x/self.Tn
        self.down2 += x/self.Tn
        self.count += 1     # 计算次数叠加

    def return_syn(self):
        if self.count != self.Tn:
            print(f"输入计算次数{self.Tn},实际计算次数{self.count}") 
        down = torch.mean(self.down1-self.down2**2)
        if down>-0.000001 and down<0.000001:
            return 1.
        up = self.up1-self.up2**2

        return up/down
    
    def reset(self):
        self.__init__(self.Tn, self.n)


# 延迟存储器
class delayer:
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
